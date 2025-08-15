--> Mall Customer Segmentation (K-Means Clustering)

This is a small project where I used K-Means clustering to split mall customers into different groups based on their age, income, and spending habits.  
The idea is simple — group similar customers together so a business can figure out who spends a lot, who doesn’t, and who might need special offers.

---

--> What This Project Does
- Loads a mall customer dataset
- Runs K-Means to create customer segments
- Uses PCA so we can see the clusters in 2D
- Shows the results with color-coded scatter plots
- Prints the Silhouette Score to check how good the clusters are

---

--> Tools & Libraries
- pandas - for working with the dataset  
- numpy - for math stuff under the hood  
- matplotlib / seaborn - for charts and plots  
- scikit-learn - for K-Means, PCA, and evaluation

---

--> Dataset Info
The dataset is Mall_Customers.csv and has:
- Customer ID  
- Gender  
- Age  
- Annual Income (in $1000s)  
- Spending Score (1–100)

Just make sure the CSV file is in the same folder as the Python script.

---

