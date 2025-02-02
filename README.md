<h1>ML with Logistic Regression to Detect Fraudulent Transactions</h1>
<h2>Dataset Info & Source</h2>
<p>This dataset is a synthetic financial dataset for fraud detection simulated by PaySim based on a sample of real mobile money transactions extracted from one month of financial logs from an African mobile money service provider. This dataset can be obtained <a href='https://www.kaggle.com/datasets/ealaxi/paysim1'>here</a>.</p>

<h2>Methods</h2>
<p>Knowing the importance of the amount of transactions in determining whether a transaction is fraudulent, I began by inspecting the summary statistics of the amount column of the dataset. This allowed me to get an idea of the distribution of the amounts.</p>
<p>I also wanted a way to more easily, and quantitatively identify outgoing transactions, so I created an additional column in the dataset titled 'isPayment' that contained a binary variable to numerically determine whether a transaction was an outgoing transaction. This column renders a 1 if the 'type' is 'PAYMENT' or 'DEBIT', and 0 otherwise.</p>
<p>I, further, created an additional column to quantitatively identify movements of money from the origin account. This column renders a 1 if the 'type' is 'CASH_OUT' or 'TRANSFER', and 0 otherwise.</p>
<p>It is important to consider the value difference of the origin and destination account when determining whether a transaction is fraudulent, as destination accounts with a vastly different value than the origin account could indicate fraud. To account for this, I created an additional column in the dataset titled 'accountDiff' that contains the absolute difference between the balance of the origin account prior to the transaction ('oldbalanceOrg'), and the tbalance of the destination account prior to the transaction ('oldbalanceDest').</p>
<p>Since these factors were now quantitative representatives most important and relevant for determining fraudulent transactions, I then created features and label variables on this data to prepare for splitting into training and testing datasets.</p>
<p>After splitting the data into training and testing datasets, I scaled the features variables to prepare for training the Logistic Regression model, fit the model on the training data, and scored the model on the training and testing data.</p>

<h2>Results</h2>
<p></p>

<h2>Libraries Used</h2>
<ul>
  <li>seaborn</li>
  <li>pandas</li>
  <li>numpy</li>
  <li>matplotlib</li>
  <li>scikit-learn</li>
</ul>
