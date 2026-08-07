#include "pch.h"
#include <iostream>
#include <random>
#include <mutex>
#include "FCN.h"

#define print(x) std::cout << x << std::endl;
#define RSEED 0
#define EPSILON pow(10, -8)
#define LEAKY_RELU_GRAD .01

#ifdef max
#undef max
#endif

RandomGen rgen = RandomGen(RSEED);

fpoint ReLU(fpoint);
fpoint d_ReLU(fpoint);
fpoint ReLU_leaky(fpoint inp);
fpoint d_ReLU_leaky(fpoint inp);

static inline fpoint _ACTIV(fpoint x) { return ReLU_leaky(x); }
static inline fpoint _D_ACTIV(fpoint x) { return d_ReLU_leaky(x); }



void printShape(Matrixd const& m) {
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



void randomise_matrix_inplace(Matrixd& m, fpoint mu = 0, fpoint sigma = .1) {
	for (int y = 0; y < m.rows(); y++) {
		for (int x = 0; x < m.cols(); x++) {
			m(y, x) = mu + sigma * (rgen.get() * 2.f - 1.f);
		}
	}
}



void he_param_init(Matrixd& m) {
	fpoint stdDev = std::sqrtf(2.f / m.cols());
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	auto dist = std::normal_distribution<fpoint>{ 0, stdDev };

	for (int y = 0; y < m.rows(); y++) {
		for (int x = 0; x < m.cols(); x++) {
			m(y, x) = dist(gen);
		}
	}
}



FCN_Layer::FCN_Layer(int inSize, int outSize, float dropOut, int activ) 
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


			if (j == 0) { print("TRAIN SET") }
			else { print("TEST SET") };
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



FCN::~FCN() {
	this->self = nullptr;
}



FCN::FCN(vectorList<int>& layerSizes, fpoint dropout,
		 vectorList<int>& activs,
		 fpoint init_mu=0, fpoint init_sigma=.1,
		 bool bias_init_0=true) {
	
	this->weights.clear(); this->bias.clear();
	this->arch.clear();

	int size = layerSizes.size();
	this->weights.resize(size - 1);
	this->bias.resize(size - 1);

	bool doRandInit = init_mu != 0 or init_sigma != 0;

	if (doRandInit) print("parameter random assignment");

	for (int i = 1; i < size; i++) {
		int size = layerSizes[i - 1];
		int nextSize = layerSizes[i];

		Matrixd& weight = this->weights[i - 1];
		weight.resize(
			nextSize, size);
		//weight.fill(initVal);
		weight.fill(0);
		if (doRandInit) {
			he_param_init(weight);
			//randomise_matrix_inplace(weight, init_mu, init_sigma);
		}
			


		Matrixd& bias = this->bias[i - 1];
		bias.resize(
			nextSize, 1);
		bias.fill(0);
		if (doRandInit and (not bias_init_0)) {
			he_param_init(bias);
			//randomise_matrix_inplace(bias, init_mu, init_sigma);
		}


		this->arch.push_back(FCN_Layer(
			size, nextSize, dropout, activs[i - 1]));
	}

	std::cout << "weights[0] abs mean = " 
		<< this->weights[0].cwiseAbs().mean()
		<< std::endl;

	//print("WEIGHTS");
	//printVector(this->weights);
	//print("BIAS");
	//printVector(this->bias);

}



void FCN::display() {
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


void softMaxMatrix_colwise_inplace(Matrixd& x) {
	// subtract max for stability, then exp all
	rVectorxd max_s = x.colwise().maxCoeff();
	// cast back into matrix, eigen likes to default to vector
	x = (x.rowwise() - max_s).array().exp().matrix();
	rVectorxd sum = x.colwise().sum();
	// specify output type, auto defaults to non compatible
	x = x.array().rowwise() / sum.array();
}



Matrixd FCN::forward(Matrixd& Xs) {

	{
		std::lock_guard<std::mutex> lock(this->paramMutex);
	
		if (Xs.rows() != this->arch[0].inSize) {

			std::cout
				<< "bad input size " << Xs.rows()
				<< " expected " << this->arch[0].inSize
				<< std::endl;
			exit(-1);
			//throw std::invalid_argument("BAD INPUT SIZE");
		}

		Matrixd x = Xs;
		int bsize = x.cols();

		for (int i = 0; i < this->arch.size(); i++) {

			auto const& W = this->weights[i];
			Matrixd B = this->bias[i].replicate(1, bsize); // copy
			bool is_last_iteration = i == this->arch.size() - 1;

			x = W * x + B;

			// last activation is softmax, 
			if (not is_last_iteration) {
				x = x.unaryExpr(&_ACTIV);
			}
			else {
				softMaxMatrix_colwise_inplace(x);
			}

		}


		return x;








	}

	

}



void d_softmax_inplace(Matrixd& X) {
	//Matrixd out;
	//out.resizeLike(X);

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


void do_dropOut_inPlace(Matrixd& X, fpoint p) {

	if (p == 0) { return; }

	//Matrixd const zeros = Matrixd::Zero(X.rows(), X.cols());
	Matrixd const ones = Matrixd::Ones(X.rows(), X.cols());
	Matrixd const zeros = 0.0 * ones;
	

	// -1 -> 1
	Matrixd mask = Matrixd::Random(X.rows(), X.cols());
	// 0 -> 1
	mask = (mask + ones) / 2.0;

	// has to be one line, as (mask > p) casts outside of Matrixd
	// without letting you cast back without static assertions
	// do not overwrite X straight away!!!
	//Matrixd& temp = zeros;
	Matrixd out = (mask.array() > p).select(X, zeros);
	
	// scale by 1-p to keep magnitudes consistent irrespective of p
	X = out / (1.f - p);
}


// TODO rewrite to use singular forward function with option to record As, Zs
// get (avg) gradient of loss w.r.t each weight, bias
vectorList<vectorList<Matrixd>>
FCN::backward(Matrixd& Xs, Matrixd& Ys) {

	std::lock_guard<std::mutex> lock(this->paramMutex);

	// values at each layer before activation
	vectorList<Matrixd> Zs;
	// .. and after
	vectorList<Matrixd> As;

	// store which values are zeroed 
	// empty = no dropout applied
	vectorList<Matrixd> dropoutMasks;

	// derivatives of loss w.r.t. weights and biases
	vectorList<Matrixd> DlDw;
	vectorList<Matrixd> DlDb;


	Matrixd X = Xs; // current layer value
	const int bsize = X.cols();

	Zs.push_back(X);
	As.push_back(X);
	// no dropout yet, append empty matrix
	// required to keep this array the correct length
	dropoutMasks.push_back(Matrixd());

	// go forward recording As, Zs and dropout masks
	for (int i = 0; i < (int)this->arch.size(); i++) {

		const auto& W = this->weights[i];
		const bool is_last_iter = i == this->arch.size() - 1;

		// copy and repeat
		Matrixd B = this->bias[i].replicate(1, bsize);

		X = W * X + B;

		Zs.push_back(X);

		// apply activation
		if (not is_last_iter) {
			X = X.unaryExpr(&_ACTIV);}
		else {
			softMaxMatrix_colwise_inplace(X);}

		// store dropout and scale by 1/(1-p) to keep
		// magnitudes consistent 
		Matrixd dropout;
		const fpoint p = this->arch[i].dropOut;
		if ((not is_last_iter) and p > 0.f) {
			// cannot do dropout in place as mask is required

			// -1 -> 1
			Matrixd rand = Matrixd::Random(X.rows(), X.cols());
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
	Matrixd error = As.back() - Ys;

	// go backwards, using stored values to backprop gradients
	for (int i = (int)this->arch.size(); i > 0; i--) {
		
		Matrixd& A_prev = As.at(i - 1);
		
		// standard gradient propagation
		Matrixd Dloss_Dbias  = error.rowwise().sum() / (fpoint)bsize;
		Matrixd Dloss_Dweight = error * A_prev.transpose() / (fpoint)bsize;
		DlDb.push_back(Dloss_Dbias);
		DlDw.push_back(Dloss_Dweight);

		// final iteration is first layer, which has no previous activation
		if (i > 1) {
			Matrixd& W = this->weights[i - 1];
			Matrixd& Z_prev = Zs.at(i - 1);	
		
			// Dloss / DA
			Matrixd grad = W.transpose() * error;
		
			Matrixd& dropout = dropoutMasks.at(i - 1);

			// if no dropout, empty matrix
			if (dropout.size() != 0) grad = grad.cwiseProduct(dropout);

			Matrixd d_activ_z = Z_prev.unaryExpr(&_D_ACTIV);
			error = grad.cwiseProduct(d_activ_z);

		}
	
	}



	// backprop records gradients backwards, so reverse
	std::reverse(DlDw.begin(), DlDw.end());
	std::reverse(DlDb.begin(), DlDb.end());

	return { DlDw, DlDb };
};




Matrixd colwiseArgmax(Matrixd& x) {
	Matrixd out;
	out.resize(1, x.cols());
	out.fill(-1);
	for (int i = 0; i < x.cols(); i++) {
		Eigen::Index max_;
		x.col(i).maxCoeff(&max_);
		
	}
	return out;
}


fpoint getAccuracy(Matrixd& A, Matrixd& B) {

	int correct = 0;
	int total = 0;
	Eigen::Index Ai, Bi;

	for (int i = 0; i < A.cols(); i++) {

		// undo ohe
		A.col(i).maxCoeff(&Ai);
		B.col(i).maxCoeff(&Bi);

		if (Ai==Bi) {
			correct++;
		}
		total++;
	}

	return (float)correct / (float)total;

}

// mse
fpoint _loss_mse(Matrixd& a, Matrixd& b) {
	return (a - b).squaredNorm() / (a.size() + EPSILON);
}


fpoint _loss_ce(Matrixd& logits, 
						  //const std::vector<int>& labels) {
						  Matrixd& Y) {

	double total_loss = -(Y.array() * logits.array().max(EPSILON).log()).sum();
	return total_loss / (fpoint)logits.cols();


}


vectorList<fpoint> FCN::test_against_unseen(int N = -1) {

	

	//return { 1, 2 };
	fpoint acc = 0, sumLoss = 0;
	int total_datapoints_tested = 0;
	int total_batches_tested = 0;
	//auto& dataset = this->trainData;
	auto& dataset = this->testData;

	for (int i = 0; i < dataset.Xs.size(); i++) {
		Matrixd& X = dataset.Xs[i];
		Matrixd& Y = dataset.Ys[i];
		Matrixd out = this->forward(X);

		
		total_batches_tested++;
		total_datapoints_tested += X.cols();
		
	
		acc += getAccuracy(out, Y);
		//sumLoss += _loss_mse(out, Y);
		sumLoss += _loss_ce(out, Y);


		if (N != -1 and total_datapoints_tested > N) break;
	}

	const fpoint n = N==-1 ? total_batches_tested : N;
	return { acc / n,  sumLoss / n };
}

// TODO replace with more efficient Fisher–Yates shuffle
void FCN::shuffle_dataset(int n = -1) {

	auto& dataset = this->trainData;

	int N = dataset.Xs.size() - 1;

	// half dataset size by default
	if (n == -1) {
		n = N;
	}
	if (n >= N) n = N;

	// todo make DataSet use better data struct for swaps
	for (int i = 0; i < n; i++) {

		int i_ = (rgen.get() * N);
		int j_ = (rgen.get() * N);

		Matrixd X = dataset.Xs[i_];
		Matrixd Y = dataset.Ys[i_];
	
		dataset.Xs[i_] = dataset.Xs[j_];
		dataset.Ys[i_] = dataset.Ys[j_];

		dataset.Xs[j_] = X;
		dataset.Ys[j_] = Y;

	
	}

}


// class to optimize array of params given gradient estimates
struct Optim {
	fpoint lr, momentum, weight_decay;
	vectorList<vectorList<Matrixd>> vels;
	//bool has_initial_vels;s;
		
	Optim(fpoint lr, fpoint momentum, fpoint weight_decay,
		  vectorList<vectorList<Matrixd>> const& sizes) {
		this->lr = lr;
		this->momentum = momentum;
		this->weight_decay = weight_decay;
		//this->has_initial_vels = false;
		this->register_vels(sizes);

		fpoint effective_lr = lr / (1 - momentum);

		// sanity check
		if (momentum != 0) {
			std::cout << std::endl
				<< "optimizer using lr = " << lr << ", momentum = "
				<< momentum << ", effective avg step size = "
				<< effective_lr << " ~ 10^" << log10f(effective_lr)
				<< std::endl;
		}
		

	}


	void register_vels(vectorList<vectorList<Matrixd>> const& sizes) {

		vels.resize(sizes.size());

		int i = 0;
		for (auto& size : sizes) {
			for (Matrixd const& m : size) {
				this->vels[i].push_back(
					Matrixd::Zero(m.rows(), m.cols()));
			}
			i++;
		
		}
		//this->has_initial_vels = true;
	}



	void optimize(int n,
				  vectorList<Matrixd>* parameters, 
				  vectorList<Matrixd>* grads) {


		if (grads->size() != parameters->size()) {
			print("BAD OPTIMIZER INPUT");
			exit(-1);
		}

		//print("INSIDE");
		//print(&(*parameters)[0](0, 0));



		//return;


		for (int i = 0; i < grads->size(); i++) {
			Matrixd& parameter = parameters->at(i);
			Matrixd& grad = grads->at(i);
			if (momentum == 0)
				parameter += grad * -this->lr;
			else {
				// update velocity
				Matrixd vel_ = 
					this->vels[n][i] * this->momentum + grad * -this->lr;
				this->vels[n][i] = vel_;
				// optimize
				parameter += vel_;


			}

			// weight decay
			if (this->weight_decay != 0) {
				parameter *= (1 - this->lr * weight_decay);
			}

				
		}
	}

};



// given an array of parameters, and having already known the 
// architecture, load parameters
void FCN::load_params(fpoint* in) {

	{

		std::lock_guard<std::mutex> lock(this->paramMutex);


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
		std::lock_guard<std::mutex> lock(this->paramMutex);

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




void FCN::train(int epochs, fpoint lr, fpoint momentum=0,
				fpoint weight_decay = 0,
				fpoint* data = nullptr, int dataN=0,
				int threads = 8) {



	Eigen::setNbThreads(threads);

	bool send_back_data = data != nullptr and dataN != 0;
	bool loopForever = epochs == -1;
	if (loopForever) epochs = 1;


	


	Optim optim(lr, momentum, weight_decay, {
		this->weights, this->bias	
	});

	fpoint SumTrainLoss = 0;

	if (send_back_data) {
		data[0] = 0;
	}

	for (int epoch = 1; loopForever or epoch < epochs + 1; 
		 epoch++) {

		// validate
		auto results = this->test_against_unseen();
		fpoint n = this->trainData.Xs.size();
		fpoint acc = results[0];
		fpoint valLoss = results[1];
		fpoint trainLoss = SumTrainLoss / n;

		if (send_back_data) {
			data[1] = 0;
			data[2] = 0;
			data[3] = trainLoss;
			data[4] = valLoss;
			data[5] = acc;
		}

		//std::cout
		//	<< "epoch " << epoch << " done\n"
		//	<< "train loss avg = " << trainLoss << "\n"
		//	<< "test loss avg = " << valLoss << "\n"
		//	<< "accuracy = " << acc << std::endl;
		//
		//print(this->weights[0].cwiseAbs().colwise().mean().mean());


		SumTrainLoss = 0;

		this->shuffle_dataset();
		

		// optimize parameters
		for (int i_data = 0; i_data < this->trainData.Xs.size(); i_data++) {

			
			//if (i_data != 0) printf("\b \b \b \b");
			//printf("%d/%d", i_data, (int)this->trainData.Xs.size());

			auto& X = this->trainData.Xs[i_data];
			auto& Y = this->trainData.Ys[i_data];


			//continue;

			auto grads = this->backward(X, Y);
			auto& D_weight = grads[0];
			auto& D_bias = grads[1];


			//continue;
			{
				std::lock_guard<std::mutex> lk(this->paramMutex);

				optim.optimize(0, &this->weights, &D_weight);
				optim.optimize(1, &this->bias, &D_bias);

			}
			
			Matrixd out = this->forward(X);
			SumTrainLoss += _loss_ce(out, Y);

			if (send_back_data){
				data[1] = (fpoint)i_data / this->trainData.Xs.size();
				data[2] = epoch;
			}
		}




	}

	// python tests for these values to tell when epochs finish
	if (send_back_data) {
		data[0] = 1;
		data[1] = 1;
	}


}











int dll_sanity_check(int x) {
	print(x);
	return x;
}








