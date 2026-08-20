<script setup>
import { computed } from 'vue'

const props = defineProps({
    data: {
        type: Array,
        default: () => []
    },
    maxheight: {
        type: Number,
        default: 40
    },
    width: {
        type: Number,
        default: 100
    },
    padding: {
        type: Number,
        default: 10
    }
})

const labelGap = 10
// allow maxheight to be height of entire element, so take away label size
const chartHeight = computed(() => props.maxheight - labelGap)
const spacing = computed(() => 
    (props.width - props.padding * 2) / props.data.length
)
const barWidth = computed(() => spacing.value * .9)
// get pixel height of bar from 0->1 value (x)
const getBarHeight = (x) => x * (chartHeight.value - props.padding)

</script>




<template>

<div class="bar-chart-container">

    <svg :viewBox="`0 0 ${width} ${maxheight}`" class="bar-chart">

        <g v-for="(item, index) in data" :key="index">

            <rect 
                class="bar"
                :x="padding + index * spacing"
                :y="chartHeight - getBarHeight(item.value)"
                :width="barWidth" :height="getBarHeight(item.value)">
            </rect>

            <text 
                class="bar-label"
                :x="padding + index * spacing + barWidth / 2"
                :y="chartHeight+10"
                text-anchor="middle"
                >
                {{ item.label }}
            </text>

        </g>

    </svg>

</div>

</template>



<style>

.bar-chart-container{
    width: 100%
}
.bar-chart {
    width: 100%;
    height: auto;
    overflow: visible;
}

.bar {
    fill: #505151;
    transition: height 0.2s ease, y 0.2s ease;
}


.bar-label {
  font-family: sans-serif;
  font-size: 8px;
  fill: #666;
}
</style>