<script setup>
import { ref, computed, onMounted } from 'vue';


const props = defineProps({
    count: {
        type: Number,
        default: 32*32
    },
    gridsize:{
        type: Number,
        default: 200
    }
});

const emit = defineEmits(['neural_network_response'])

const length = Math.sqrt(props.count);
const isMicroserviceUp = ref(false)

// has to be generated with a function as it depends on props
function generate_grid_array(count){
    return Array.from({length: count}, 
        (_, i) => ({
            id:i, 
            x:i % length, y:Math.trunc(i / length),
            on:false
    }))
}
const array = ref(generate_grid_array(props.count));

// size of each cell in pixels
const pxSize = computed(() => `${props.gridsize / length}px`);
const drawRadius = 1;
const isMouseDown = ref(false)

function mouseDown(){
    isMouseDown.value = true; }
function mouseUp(){
    isMouseDown.value = false; }


// access Python and C++ backend to get a prediction
async function usePythonMicroservice(data){
    let toReturn = null;

    try{
        const res = await fetch('http://127.0.0.1:8000/api/get_prediction', {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({grid : data})
        });
        toReturn = await res.json()
    }
    catch (error) {
        console.error('microservice error:', error.message)
    }
    return toReturn
}

// sync data with Bars.vue using emit
async function syncData(){
    
    const response = await getData()

    if (response == null) {
        isMicroserviceUp.value = false;
        return;
    }
    
    isMicroserviceUp.value = true;
    emit('neural_network_response', Array.from(response.distribution))

}

// stack all the functions required to read grid pixels,
// scale to required size, communicate with backend and return
// distribution predicted
async function getData(){
    return await 
        usePythonMicroservice(
            scaleToMNIST(
                array.value.map(cell => Number(cell.on))))
}

// grid can be any size but MNIST requires 28*28 grid, so 
// nearest neighbour interpolate to 28*28
function scaleToMNIST(data){
    const n = length
    let minX=n, minY=n, maxX=-1, maxY=-1;

    // find bounding box
    for (let y = 0; y < n; y++){
    for (let x = 0; x < n; x++){
        if (data[y*n + x] > 0){
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y; }
    }}

    // handle case where nothing has been drawn
    if (maxX === -1) return Array.from({length: 28*28},
        (element, index) => (0)
    );

    // expand by 4 on each size, as mnit images tend to have about
    // 4 pixels of empty space before the edge of the image
    const startX = Math.max(0, minX - 4);
    const startY = Math.max(0, minY - 4);
    const endX = Math.min(n-1, maxX + 4);
    const endY = Math.min(n-1, maxY + 4);

    const cropX = endX - startX + 1;
    const cropY = endY - startY + 1;

    const mnistOUT = new Float32Array(28 * 28);

    // nearest neighbour mapping
    for (let y = 0; y < 28; y++){
    for (let x = 0; x < 28; x++){
        const X = startX + Math.floor((x/28)*cropX)
        const Y = startY + Math.floor((y/28)*cropY)
        mnistOUT[y*28 + x] = data[Y*n + X]
    }}

    // console.log(mnistOUT.length)

    return Array.from(mnistOUT)

}


// handle mouse passing over cell event
function mouseOverCell(cell){

    if (isMouseDown.value){

        // find bounding box to avoid iterating each cell
        const rad = drawRadius
        const minX = Math.max(0, cell.x - rad);
        const maxX = Math.min(length - 1, cell.x + rad);
        const minY = Math.max(0, cell.y - rad);
        const maxY = Math.min(length - 1, cell.y + rad);

        const grid = array.value

        // let i = 0;
        for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
            // i++
            const dx = x - cell.x;
            const dy = y - cell.y;

            const dxy = dx * dx + dy * dy
            const drawradsq = drawRadius * drawRadius

            // actually perform a circular brush test
            if (dxy <= drawradsq){
                const index = y * length + x;
                grid[index].on = 1
            }


        }}

        // console.log(i)



            

    }
}

// reset grid
function reset(){
    array.value.forEach(element => {
        element.on = false
    });
}

// timer to call microservice a few times a second
let timer = null;
onMounted(() => {
    timer = setInterval(() => {
        syncData();
    }, 200
)})

</script>



<template>

<div class="centered">
<div
    class="grid",
    :style="{
        gridTemplateColumns: `repeat(${length}, ${pxSize})`,
        gridTemplateRows: `repeat(${length}, ${pxSize})`,
        width: `${gridsize}px`,
        height: `${gridsize}px`
    }"
    @mousedown="mouseDown()"
    @mouseup="mouseUp()"
    @mouseleave="mouseUp()"
    >
        <div
            v-for="cell in array"
            :key="cell.id"
            class="cell"
            :class="{pressed : cell.on }"
            @mouseenter="mouseOverCell(cell)"
            
            ></div>
</div>


</div>

<div class="status" :class="{ offline: !isMicroserviceUp }">
    <span class="status-dot"></span>
    {{ isMicroserviceUp ? 'Connected' : 'Offline' }}
</div>


<div class="centered">
<button @click="reset()">
    RESET
</button>
</div>



<!-- <div v-if="isMicroserviceUp">
    <p>
        MICROSERVICE WORKING
    </p>
</div>

<div v-else>
    <p>
        MICROSERVICE NOT WORKING
    </p>
</div> -->


</template>



<style scoped>

.status {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    justify-content: center;
    font-size: 0.75rem;
    color: #0ca855;
    margin-top: 0.5rem;
}

.status.offline {
    color: #403f3fbd;
}

.status-dot {
    width: 8px;
    height: 8px;
    background: #0ca855;
}

.status.offline .status-dot {
    background: #403f3fbd;
}



.centered{
    display: flex;
    justify-content: center;
    align-items: center;
}


.grid{
    display: grid;
    border: 1px solid #333;
    user-select: none;
}

.cell{
    border: 0px solid #ccc;
    box-sizing: border-box;
    cursor: pointer;
    background-color: #fff;
    user-select: none;
}

.cell:hover { 
    background-color: #eee; 
    user-select: none;
}

.cell.pressed {
    background-color: #333;
    user-select: none;
}

button{
  padding: 0.4rem 1rem;
  border: 1px solid #ccc;
  background: #696868;
  color: white;
  font-size: 0.85rem;
  cursor: pointer;
  border-radius: 3px;
}

button:hover{
    color: grey;
    background-color: #eee;
}



</style>