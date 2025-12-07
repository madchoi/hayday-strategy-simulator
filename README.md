# Hay Day Strategy Simulator 🚜🌾👨‍🌾

## 📌 Project Overview  
This project successfully developed and deployed a strategic simulation tool for the mobile game **Hay Day**. The focus was on leveraging mathematical optimization to transcend simple efficiency metrics and generate prescriptive, optimal production plans for players across all experience levels.

The core result is an interactive Streamlit application that maximizes total XP gain under real-world game constraints.

## 🎯 Key Achievements & Results  
- **Data Engineering & Stabilization:** Developed a robust **Pandas/BeautifulSoup** data pipeline to scrape raw production data from the Hay Day Wiki. This pipeline included critical data-driven fixes to stabilize unreliable time and resource inputs (e.g., correcting zero-time items like Ores and Honeycomb).
- **Advanced Optimization (MILP):** Designed and implemented a **Mixed-Integer Linear Program (MILP)** using **PuLP** to solve for maximum XP output. This involved complex modeling of material flow and consumption.
- **Constraint Resolution:** Successfully debugged and resolved repeated "Unbounded" solver errors by engineering a hybrid constraint system. This system uses both Equality and Inequality constraints to correctly model Initial Stock and prevent the exploitation of low-cost ingredients.
- **Realistic Simulation:** Integrated the game's core resource limitations by implementing the Machine Capacity Constraint, ensuring the model generates a diverse, realistic, and truly optimal simultaneous production plan.
- **Deployment:** Created an interactive, web-ready **Streamlit** application with a custom theme and filtering, allowing users to generate optimal plans instantly based on level and time window

## 🛠 Tools & Libraries  
- **Optimization:** PuLP (for MILP modeling and solving)
- **Web App:** Streamlit (for interactive deployment and custom UI)
- **Data Engineering:** Python, Pandas, NumPy (for data wrangling, cleaning, and model input preparation)
- **Scraping & Visualization:** BeautifulSoup, Matplotlib (for data acquisition and results visualization)

## 📂 Project Structure  
<pre>
hayday-strategy-simulator/
  |── app/ 
    |── app.py             # The Streamlit application entry point (main script)
    |── requirements.txt   # Deployment dependencies (Pandas, PuLP, Streamlit)
    |── goods_final.csv    # Final, clean dataset used by app.py
  |── data/ # scraping, cleaning, and structuring data
  ├── notebooks/ # Jupyter notebooks for exploration + analysis
  └── README.md # project description
</pre>

## ✨ Technical Wins & Key Takeaways
- Mastered the end-to-end process of translating real-world business rules into a functional mathematical optimization model
- Gained deep experience in diagnosing and solving complex LP/MILP solver failures (specifically Unbounded and Infeasible states) through structural constraint modifications
- Improved skills in deployment, custom CSS theming, and designing robust, cached web applications

---

👩‍💻 *Created as part of the Data Science Union Second Quarter Curriculum Project*  
