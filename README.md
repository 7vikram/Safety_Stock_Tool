# Safety Stock Tool Recommendation Model

## Overview

The **Safety Stock Tool Recommendation Model** is a Python-based tool designed for supply chain inventory management. It calculates key inventory metrics such as safety stock, reorder point, and stockout probabilities, and provides a recommendation for optimal safety stock levels based on historical consumption, future demand, lead time, and service levels.

The tool is integrated with a frontend built using **Tailwind CSS** and rendered using **Plotly.js** for interactive graphs, providing visual insights into inventory levels, safety stock, and replenishment cycles.

## Demo

![Safety Stock Tool Demo](/static/SS_Tool_Demo.png)

## Features

- **Safety Stock Calculation**: Computes safety stock based on a desired service level, lead time, and historical consumption.
- **Reorder Point Calculation**: Determines the reorder point based on safety stock and demand.
- **Stockout Probability**: Calculates the probability of stockouts for different levels of safety stock.
- **Future Demand**: Input of future demand to simulate inventory replenishment.
- **Replenishment Simulation**: Simulates inventory replenishment based on the minimum order quantity (MOQ) and lead time.
- **Interactive Visualization**: Visualizes the inventory trajectory, safety stock levels, and replenishment cycles using Plotly graphs.

## Requirements

- Python 3.8+
- Pandas
- NumPy
- SciPy
- Plotly
- Tailwind CSS (for frontend)
- Flask (for backend, if needed)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/7vikram/Safety_Stock_Tool.git
   cd Safety_Stock_Tool
