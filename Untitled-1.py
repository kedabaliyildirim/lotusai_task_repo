
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pushbullet import Pushbullet
from dotenv import load_dotenv
import os

load_dotenv()
try:
    pb = Pushbullet(os.getenv('PUSHBULLET_API_KEY'))
except:
    pb = None


# Read
df = pd.read_excel('Online_Retail.xlsx')
df.info()
df.head()


# Preprocess
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

df = df[~df['Description'].str.contains('Adjust bad debt', case=False, na=False)]



# Group by
purchase_group = df.groupby('InvoiceNo')['Description'].apply(list).reset_index()
purchase_group



try:
        
    transactions = purchase_group['Description'].apply(lambda x: [str(item) for item in x if pd.notnull(item)]).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)

    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)


    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    print(rules.head())
except Exception as e:
    print(e)
    if pb:
        pb.push_note('Error', str(e))
    raise e



try:
    push = pb.push_note("Apriori", "Done")
except Exception as e:
    print("Error sending pushbullet notification")

frequent_itemsets.to_csv('frequent_itemsets.csv', index=False)

rules.to_csv('association_rules.csv', index=False)



top_itemsets = frequent_itemsets.nlargest(10, 'support')

plt.figure(figsize=(10, 6))
sns.barplot(x='support', y='itemsets', data=top_itemsets)
plt.title('Top 10 Frequent Itemsets by Support')
plt.xlabel('Support')
plt.ylabel('Itemsets')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.title('Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()


itemsets_matrix = df_encoded.T.dot(df_encoded)
sns.heatmap(itemsets_matrix, cmap="YlGnBu")
plt.title('Heatmap of Itemset Correlations')
plt.show()


