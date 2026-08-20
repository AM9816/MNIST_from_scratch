<script setup>
import { ref } from 'vue'

import MNIST_grid from './Grid.vue';
import Bars from './Bars.vue';

const predData = ref(Array.from({length: 10},
  (element, index) => ({
    label: index, 
    value: 1
  })
))

function handle_pred_data(data){
  predData.value = predData.value.map(
    (item, index) => ({
      ...item,
      value: data[index]
    })
  )
}
</script>


<template>
  <div class="app-shell">
    <h1 class="app-title">Digit Recognizer no Machine Learning Libraries</h1>
  <MNIST_grid @neural_network_response="handle_pred_data">
  </MNIST_grid>
  <Bars :data="predData"></Bars>
  </div>
</template>


<style scoped>
.app-shell {
  max-width: 500px;
  margin: 2rem auto;
  padding: 0 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.app-title {
  font-family: Georgia, serif;
  font-size: 1.4rem;
  font-weight: 500;
  color: #2a2a2a;
  text-align: center;
  margin: 0;
}
</style>