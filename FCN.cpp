#include "pch.h"
#include "FCN.h"


#ifdef max
#undef max
#endif

RandomGen<fpoint> rgen(nn::rSeed);


void printShape(Matrix const& m) {
	std::cout 
		<< "(" << m.rows()
		<< ", " << m.cols() 
		<< ")" << std::endl;
}



template<typename T>
void printVector(vectorList<T>& l) {
	for (auto& e : l) { std::cout << e << ", "; };
	std::cout << std::endl;
}


// basic uniform distribution initialization
void uniformParamInit(Matrix& m, fpoint mu = 0, fpoint sigma = .1) {
	for (int y = 0; y < m.rows(); y++) {
		for (int x = 0; x < m.cols(); x++) {
			m(y, x) = mu + sigma * (rgen.get() * 2.f - 1.f);
		}
	}
}


// Kaiming parameter initialization
// https://www.geeksforgeeks.org/deep-learning/kaiming-initialization-in-deep-learning/
void heParamInit(Matrix& m) {

	std::mt19937& engine = rgen.engine;
	auto distFunc = std::normal_distribution<fpoint>{ 
		0, std::sqrtf(2.f / m.cols()) };

	for (int y = 0; y < m.rows(); y++) {
		for (int x = 0; x < m.cols(); x++) {
			m(y, x) = distFunc(engine);
		}
	}
}


FCN_Layer::FCN_Layer(int inSize, int outSize, 
					 float dropOut, nn::Activation activ) 
	: inSize(inSize), outSize(outSize), dropOut(dropOut), activation(activ)
{ }



void FCN::display_dataset(bool print_actual_data=false) {

	if (this->datapointLength == -1) {
		print("no datapoints registered");
		return;
	}

	if (print_actual_data) {
		for (int j = 0; j < 2; j++) {
			auto& dataset = j == 0 ?
				this->trainData : this->testData;


			if (j == 0) { print("TRAIN SET"); }
			else { print("TEST SET"); };
			for (int i = 0; i < dataset.Xs.size(); i++) {
				auto& x = dataset.Xs[i];
				auto& y = dataset.Ys[i];
				print("batch nr " + std::to_string(i));
				print("X"); print(x);
				print("Y"); print(y);
			}
		}
	}


	auto& dataset = this->trainData.Xs;
	int bsize = dataset[0].size() / this->datapointLength;
	int totalDatapoints = bsize * (dataset.size() - 1) + 
						   (dataset.back().size() / this->datapointLength);
	std::cout
		<< "train dataset\n" << dataset.size()
		<< " batches of size " << bsize << std::endl
		<< totalDatapoints << " total datapoints" 
		<< std::endl << std::endl;

	auto& dataset2 = this->testData.Xs;
	bsize = dataset2[0].size() / this->datapointLength;
	totalDatapoints = bsize * (dataset2.size() - 1) +
		(dataset2.back().size() / this->datapointLength);
	std::cout
		<< "test dataset\n" << dataset2.size()
		<< " batches of size " << bsize << std::endl
		<< totalDatapoints << " total datapoints"
		<< std::endl << std::endl;

}



//FCN::~FCN() {
//	this->self = nullptr;
//}



FCN::FCN(vectorList<int>& layerSizes, fpoint dropout,
		 vectorList<nn::Activation>& activs,
		 bool bias_init_0=true) {
	
	this->weights.clear(); this->bias.clear();
	this->arch.clear();

	int size = layerSizes.size();
	this->weights.resize(size - 1);
	this->bias.resize(size - 1);

	for (int i = 1; i < size; i++) {
		int size = layerSizes[i - 1];
		int nextSize = layerSizes[i];

		Matrix& weight = this->weights[i - 1];
		weight.resize(
			nextSize, size);

		weight.fill(0);
		heParamInit(weight);

			


		Matrix& bias = this->bias[i - 1];
		bias.resize(
			nextSize, 1);
		bias.fill(0);
		if (not bias_init_0) {
			heParamInit(bias);
		}
		


		this->arch.push_back(FCN_Layer(
			size, nextSize, dropout, activs[i - 1]));
	}

	std::cout << "weights[0] abs mean = " 
		<< this->weights[0].cwiseAbs().mean()
		<< std::endl;


}



void FCN::display_architecture() {
	std::cout << "FCN with " << this->arch.size() << " layers" << std::endl;
	int i = 1;
	for (FCN_Layer const &layer : this->arch) {
		std::cout 
			<< "layer " << i
			<< ", in = " << layer.inSize
			<< ", out = " << layer.outSize
			<< ", dropout = " << layer.dropOut
			<< std::endl;
		i++;
	}

}


void softmax_colwise(Matrix& x) {
	// subtract max for stability, then exp all
	rVectorxd max_s = x.colwise().maxCoeff();
	// cast back into matrix, eigen likes to default to vector
	x = (x.rowwise() - max_s).array().exp().matrix();
	rVectorxd sum = x.colwise().sum();
	// specify output type, auto defaults to non compatible
	x = x.array().rowwise() / sum.array();
}

// efficient eigen implementations of relu and leaky relu .. 
Matrix eval_activation(Matrix& x, nn::Activation activ) {
	
	switch (activ) {
	case nn::Activation::RELU:
		return x.cwiseMax(0.0);
	case nn::Activation::LEAKY_RELU:
		return x.array().max(x.array() * nn::LEAKY_RELU_GRAD);
	case nn::Activation::IDENTITY:
		return x;
	}
}
// .. and their derivatives
Matrix eval_activation_derivative(Matrix& x, nn::Activation activ) {
	switch (activ) {
	case nn::Activation::RELU:
		return (x.array() > 0.0).cast<fpoint>();
	case nn::Activation::LEAKY_RELU:
		return (x.array() > 0.0).select(
			Matrix::Constant(x.rows(), x.cols(), 1.f),
			Matrix::Constant(x.rows(), x.cols(), nn::LEAKY_RELU_GRAD));
	case nn::Activation::IDENTITY:
		return Matrix::Ones(x.rows(), x.cols());
	}
}

// regular forward pass without gradient tracking
Matrix FCN::forward(Matrix& Xs) {

	{
		std::lock_guard<std::mutex> lock(this->paramConcurrentLock);
	
		if (Xs.rows() != this->arch[0].inSize) {

			std::cout
				<< "bad input size " << Xs.rows()
				<< " expected " << this->arch[0].inSize
				<< std::endl;
			//exit(-1);
			//throw std::invalid_argument("BAD INPUT SIZE");
			return Matrix();
		}

		Matrix x = Xs;
		int bsize = x.cols();

		for (int i = 0; i < this->arch.size(); i++) {

			bool isLastIter = i == this->arch.size() - 1;

			auto const& W = this->weights[i];

			// slow, allocates per iteration
			//Matrix B = this->bias[i].replicate(1, bsize); 
			//x = W * x + B;

			// equivalent, but faster
			x = (W * x).colwise() + this->bias[i].col(0);

			// last activation is softmax, 
			if (not isLastIter) {
				//x = x.unaryExpr(&_ACTIV);
				//x = x.unaryExpr(_activ);
				x = eval_activation(x, this->arch[i].activation);
			}
			// softmax final for classification
			else {
				softmax_colwise(x);
			}
		}


		return x;
	}
}


// mse
// training is done with cross entropy loss, this is just 
// for sanity checks 
fpoint loss_mse(Matrix& a, Matrix& b) {
	return (a - b).squaredNorm() / (a.size() + nn::EPSILON);
}


// cross entropy loss
fpoint loss_ce(Matrix& logits,
	//const std::vector<int>& labels) {
	Matrix& Y) {

	fpoint total_loss = -(Y.array() * logits.array().max(nn::EPSILON).log()).sum();
	return total_loss / (fpoint)logits.cols();


}


void softmax_derivative(Matrix& X) {

	for (int i = 0; i < X.cols(); i++) {
		//fpoint sigma = X.col(i).array().exp().sum();
		fpoint sigma = X.col(i).array().exp().sum();
		for (int j = 0; j < X.rows(); j++) {
			fpoint sm = std::exp(X(j, i)) / sigma;
			X(j, i) = sm * (1.f - sm);
		}
	}

	//return out;


}



// get (avg) gradient of loss w.r.t each weight, bias
// also return loss to avoid recomputation later in training
vectorList<vectorList<Matrix>>
FCN::backward(Matrix& Xs, Matrix& Ys,
			  bool returnLoss = false) {

	std::lock_guard<std::mutex> lock(this->paramConcurrentLock);

	// values at each layer before activation ..
	vectorList<Matrix> Zs;
	// .. and after
	vectorList<Matrix> As;

	// store which values are zeroed 
	// empty = no dropout applied
	vectorList<Matrix> dropoutMasks;

	// derivatives of loss w.r.t. weights and biases
	vectorList<Matrix> DlDw;
	vectorList<Matrix> DlDb;

	// pre allocate to avoid allocation per call
	auto n = this->arch.size();
	Zs.reserve(n); As.reserve(n);
	dropoutMasks.reserve(n);
	DlDb.reserve(n); DlDb.reserve(n);


	Matrix X = Xs; // current layer value
	const int bsize = X.cols();

	Zs.push_back(X);
	As.push_back(X);
	// no dropout yet, append empty matrix
	// required to keep this array the correct length
	dropoutMasks.push_back(Matrix());

	// go forward recording As, Zs and dropout masks
	for (int i = 0; i < (int)this->arch.size(); i++) {

		const auto& W = this->weights[i];
		const bool is_last_iter = i == this->arch.size() - 1;

		// copy and repeat
		Matrix B = this->bias[i].replicate(1, bsize);

		X = W * X + B;

		Zs.push_back(X);

		// apply activation
		if (not is_last_iter) {
			//X = X.unaryExpr(&_ACTIV);}
			X = eval_activation(X, this->arch[i].activation);
		}
		else {
			softmax_colwise(X);}

		// store dropout and scale by 1/(1-p) to keep
		// magnitudes consistent 
		Matrix dropout;
		const fpoint p = this->arch[i].dropOut;
		if ((not is_last_iter) and p > 0.f) {
			// cannot do dropout in place as mask is required

			// -1 -> 1
			Matrix rand = Matrix::Random(X.rows(), X.cols());
			rand = (rand.array() + 1.f) / 2.f; // 0 -> 1
			dropout = ((rand.array() > p)).cast<fpoint>();

			// scale to maintain magnitudes
			dropout = (dropout / (1.f - p)).matrix();

			// apply dropout
			X = X.cwiseProduct(dropout);
		}
		
		dropoutMasks.push_back(dropout);

		// X now stores post activation and dropout
		As.push_back(X);

	}


	// correction required at each layer
	// dloss / dZ = A - y (with cross entropy loss)
	Matrix error = As.back() - Ys;

	//Matrix finalOut = As.back();

	// go backwards, using stored values to backprop gradients
	for (int i = (int)this->arch.size(); i > 0; i--) {
		
		Matrix& A_prev = As.at(i - 1);
		
		// standard gradient propagation
		Matrix Dloss_Dbias  = error.rowwise().sum() / (fpoint)bsize;
		Matrix Dloss_Dweight = error * A_prev.transpose() / (fpoint)bsize;
		DlDb.push_back(Dloss_Dbias);
		DlDw.push_back(Dloss_Dweight);

		// final iteration is first layer, which has no previous activation
		if (i > 1) {
			Matrix& W = this->weights[i - 1];
			Matrix& Z_prev = Zs.at(i - 1);	
		
			// Dloss / DA
			Matrix grad = W.transpose() * error;
		
			Matrix& dropout = dropoutMasks.at(i - 1);

			// if no dropout, empty matrix
			if (dropout.size() != 0) grad = grad.cwiseProduct(dropout);

			//Matrix d_activ_z = Z_prev.unaryExpr(_d_activ);
			Matrix d_activ_z = eval_activation_derivative(Z_prev, arch[i - 2].activation);
			error = grad.cwiseProduct(d_activ_z);

		}
	
	}


	// backprop records gradients backwards, so reverse
	std::reverse(DlDw.begin(), DlDw.end());
	std::reverse(DlDb.begin(), DlDb.end());

	if (returnLoss) {
		fpoint loss = loss_ce(As.back(), Ys);
		return { DlDw, DlDb, { Matrix::Constant(1, 1, loss)} };
	}
	else {
		return { DlDw, DlDb };
	}

};





fpoint get_accuracy(Matrix& A, Matrix& B) {

	int correct = 0;
	int total = 0;
	Eigen::Index Ai, Bi;

	for (int i = 0; i < A.cols(); i++) {

		// undo one hot encoding
		A.col(i).maxCoeff(&Ai);
		B.col(i).maxCoeff(&Bi);

		if (Ai==Bi) {
			correct++;
		}
		total++;
	}

	return (float)correct / (float)total;

}





// holdout verification function, 
// N = -1 iterates over entire test set
vectorList<fpoint> FCN::test_against_unseen(int N = -1) {


	fpoint correct = 0, sumLoss = 0;
	int datapointsTested = 0;
	
	//auto& dataset = this->trainData;
	auto& dataset = this->testData;

	for (int i = 0; i < dataset.Xs.size(); i++) {

		Matrix& X = dataset.Xs[i];
		Matrix& Y = dataset.Ys[i];
		Matrix out = this->forward(X);

		
		const fpoint bsize = X.cols();

		// scale by amount of datapoints in batch to avoid
		// higher weight for smaller batches towards the end
		correct += get_accuracy(out, Y) * bsize;
		sumLoss += loss_ce(out, Y) * bsize;
		
		datapointsTested += (int)bsize;
		
		//sumLoss += loss_mse(out, Y);
		

		if (N != -1 and datapointsTested >= N) break;
	}

	const fpoint n = datapointsTested;
	
	// return impossible value if dataset does not exist
	if (n == 0.f) return { -1.f, -1.f };
	return { correct / n,  sumLoss / n };
}

// basic shuffle
// TODO replace with more efficient Fisher–Yates shuffle
//void FCN::shuffle_dataset(int n = -1) {
//
//	auto& dataset = this->trainData;
//
//	int N = dataset.Xs.size();
//
//	// half dataset size by default
//	if (n == -1) {
//		n = N;
//	}
//	if (n >= N) n = N;
//
//	// todo make DataSet use better data struct for swaps
//	for (int i = 0; i < n; i++) {
//
//		int i_ = (rgen.get() * N);
//		int j_ = (rgen.get() * N);
//
//		Matrix X = dataset.Xs[i_];
//		Matrix Y = dataset.Ys[i_];
//	
//		dataset.Xs[i_] = dataset.Xs[j_];
//		dataset.Ys[i_] = dataset.Ys[j_];
//
//		dataset.Xs[j_] = X;
//		dataset.Ys[j_] = Y;
//
//	
//	}
//
//}


// full, efficient per batch shuffle
// Fisher - Yates shuffle
// https://www.geeksforgeeks.org/dsa/shuffle-a-given-array-using-fisher-yates-shuffle-algorithm/
void FCN::shuffle_dataset(int n = -1) {

	auto& dataset = this->trainData;
	int N = dataset.Xs.size();

	// store indexes per datapoint
	vectorList<vectorList<int>> indexes;
	for (int batch = 0; batch < N; batch++) {
		for (int i = 0; i < dataset.Xs[batch].cols(); i++) {
			indexes.push_back({
				batch, i	
			}); 
	}}

	int dps = indexes.size();
	int swaps = (n == -1 or n > dps - 1) ? dps - 1 : n;

	Vectorxd xTmp, yTmp;

	for (int
		i = dps - 1,
		done = 0;

		i > 0 and done < swaps;

		i--, done++) {

		const int j = rgen.randint(0, i);

		if (i != j) {

			auto const& X1_i = indexes[i];
			auto const& X2_i = indexes[j];

			
			xTmp = dataset.Xs[X1_i[0]].col(X1_i[1]);
			dataset.Xs[X1_i[0]].col(X1_i[1]) =
				dataset.Xs[X2_i[0]].col(X2_i[1]);
			dataset.Xs[X2_i[0]].col(X2_i[1]) = xTmp;

			yTmp = dataset.Ys[X1_i[0]].col(X1_i[1]);
			dataset.Ys[X1_i[0]].col(X1_i[1]) =
				dataset.Ys[X2_i[0]].col(X2_i[1]);
			dataset.Ys[X2_i[0]].col(X2_i[1]) = yTmp;





		}

	}
	


}


// class to optimize array of params given gradient estimates
struct Optim {

	fpoint lr, momentum, weightDecay;
	vectorList<vectorList<Matrix>> vels;
	//bool has_initial_vels;s;
		
	Optim(fpoint lr, fpoint momentum, fpoint weightDecay,
		  //vectorList<vectorList<Matrix>> const& sizes) {
		  // list of pointers to avoid copies
		  vectorList<const vectorList<Matrix>*> params) {
		
		this->lr = lr;
		this->momentum = momentum;
		this->weightDecay = weightDecay;
		//this->has_initial_vels = false;
		this->register_vels(params);

		fpoint stepSize = lr / (1 - momentum);

		// sanity check
		if (momentum != 0) {
			std::cout << std::endl
				<< "optimizer using lr = " << lr << ", momentum = "
				<< momentum << ", effective step size = "
				<< stepSize << " ~ 10^" << log10f(stepSize)
				<< std::endl;
		}
		

	}

	// add zero matrices of the correct shape to 
	// internal velocity vectors
	void register_vels(vectorList<const vectorList<Matrix>*> params) {

		vels.resize(params.size());

		int i = 0;
		for (auto& p : params) {
			for (Matrix const& m : *p) {
				this->vels[i].push_back(
					Matrix::Zero(m.rows(), m.cols()));
			}
			i++;
		
		}
		//this->has_initial_vels = true;
	}


	// 
	void optimize(int velocityIndex,
				  vectorList<Matrix>* parameters, 
				  vectorList<Matrix>* grads) {


		if (grads->size() != parameters->size()) {
			print("BAD OPTIMIZER INPUT");
			return;
			//exit(-1);
		}

		//print("INSIDE");
		//print(&(*parameters)[0](0, 0));



		//return;

		auto& velocityArray = this->vels[velocityIndex];

		for (int i = 0; i < grads->size(); i++) {
			Matrix& parameter = parameters->at(i);
			Matrix& grad = grads->at(i);
			if (momentum == 0)
				parameter += grad * -this->lr;
			else {
				// update velocity
				Matrix vel = 
					velocityArray[i] * this->momentum + grad * -this->lr;
				velocityArray[i] = vel;
				// optimize
				parameter += vel;


			}

			// weight decay (analytic)
			if (this->weightDecay != 0) {
				parameter *= (1 - this->lr * weightDecay);
			}

				
		}
	}

};



// given an array of parameters, and having already known the 
// architecture, load parameters
void FCN::load_params(fpoint* in) {

	{

		std::lock_guard<std::mutex> lock(this->paramConcurrentLock);


		
		fpoint* dataptr = in;
		//int i = 0;
		for (auto& weight : this->weights) {
			for (auto& val : weight.reshaped()) {
				val = *dataptr; dataptr++;
				//i++;
			}
		}
		for (auto& bias : this->bias) {
			for (auto& val : bias.reshaped()) {
				val = *dataptr; dataptr++;
				//i++;
			}
		}
		//print(i);
	}
	
	}



// reduce parameters down to 1D list to pass off to python
void FCN::serialize(fpoint* out) {
	{
		std::lock_guard<std::mutex> lock(this->paramConcurrentLock);

		fpoint* dataptr = out;
		//int i = 0;
		for (auto& weight : this->weights) {
			for (auto val : weight.reshaped()) {
				*dataptr = val; dataptr++;
				//i++;
			}
		}
		for (auto& bias : this->bias) {
			for (auto val : bias.reshaped()) {
				*dataptr = val; dataptr++;
				//i++;
			}
		}
	}

}



// epochs = -1 trains forever
void FCN::train(int epochs, fpoint lr, fpoint momentum=0,
				fpoint weightDecay = 0,
				fpoint* data = nullptr, int dataN=0,
				int threads = 8, bool printStatsPerEpoch=true) {


	// allow large matrix operations to be parallelized
	// requires openmp
	Eigen::setNbThreads(threads);

	bool relayDataToPython = data != nullptr and dataN != 0;
	bool loopForever = epochs == -1;
	if (loopForever) epochs = 1;


	


	Optim optim(lr, momentum, weightDecay, {
		&this->weights, &this->bias	
	});

	fpoint SumTrainLoss = 0;

	if (relayDataToPython) {
		data[0] = 0;
	}

	for (int epoch = 1; loopForever or epoch < epochs + 1; 
		 epoch++) {

		// validate
		auto results = this->test_against_unseen();
		fpoint acc = results[0];
		fpoint valLoss = results[1];
		fpoint trainLoss = SumTrainLoss / this->trainData.Xs.size();

		if (relayDataToPython) {
			data[1] = 0;
			data[2] = 0;
			data[3] = trainLoss;
			data[4] = valLoss;
			data[5] = acc;
		}

		if (printStatsPerEpoch) {
			std::cout << std::endl // extra endl to get past tqdm bar 
				<< "epoch " << epoch << " starting\n"
				<< "train loss avg = " << trainLoss << "\n"
				<< "test loss avg = " << valLoss << "\n"
				<< "accuracy = " << acc << std::endl;
		}



		SumTrainLoss = 0;

		this->shuffle_dataset();
		

		// optimize parameters
		for (int i_data = 0; i_data < this->trainData.Xs.size(); i_data++) {

			auto& X = this->trainData.Xs[i_data];
			auto& Y = this->trainData.Ys[i_data];


			//continue;

			auto grads = this->backward(X, Y, true);
			auto& D_weight = grads[0];
			auto& D_bias = grads[1];
			fpoint trainLoss = grads[2][0](0, 0);



			//continue;
			{
				std::lock_guard<std::mutex> lk(this->paramConcurrentLock);

				optim.optimize(0, &this->weights, &D_weight);
				optim.optimize(1, &this->bias, &D_bias);

			}
			
			//Matrix out = this->forward(X);
			//SumTrainLoss += loss_ce(out, Y);
			SumTrainLoss += trainLoss;

			if (relayDataToPython){
				data[1] = (fpoint)i_data / this->trainData.Xs.size();
				data[2] = epoch;
				data[3] = SumTrainLoss / (1.f + i_data);
			}
		}




	}

	// python tests for these values to tell when epochs finish
	if (relayDataToPython) {
		data[0] = 1;
		data[1] = 1;
	}


}











int dll_sanity_check(int x) {
	print(x);
	return x;
}








