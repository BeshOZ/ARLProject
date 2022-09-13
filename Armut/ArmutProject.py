## Association Rule Based Recommender System

#########
# imports and settings
#########
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 800)

pd.set_option('display.expand_frame_repr', False)


#########
# Prepare the data.
#########

ar_data = pd.read_csv('armut_data.csv')
ar_data.head()

#########
# Make a new feature called Hizmet that consists of Category ID and Service ID
#########
ar_data["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in ar_data.values]

#########
# In order to apply association rule learning we need to get the data into a specific format
# this format should consist a basket of products, since we don't have that in this dataset
# we will make a new variable that will represents the basket IDs
#########
ar_data["CreateDate"] = pd.to_datetime(ar_data["CreateDate"])
ar_data["NEW_DATE"] = ar_data["CreateDate"].dt.strftime("%Y-%m")

ar_data["SepetID"] = ar_data["UserId"].astype(str) + "_" + ar_data["NEW_DATE"].astype(str)

#########
# In this step, we will make the association rules
#########

# the pivot table:
arTable = ar_data.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
arTable.head()
#Creating the rules
frequent_itemsets = apriori(arTable,min_support=0.01,use_colnames=True)
rules = association_rules(frequent_itemsets,metric="support",min_threshold=0.01)


# Now that we have the rules table we can make predictions based on either the support values, confidence values, lift
# values or a combo of them.


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

# By using the function defined above we can easily call this function with a service input to get recommendations to
# that service.



