import pandas as pd
import os

images = pd.read_csv("sn8_data_val.csv")
prefixes = images[images.columns[0]].tolist()
image_names = [os.path.basename(prefix)[:-4] for prefix in prefixes]

df = pd.read_csv("Louisiana-East_Training_Public/Louisiana-East_Training_Public_reference.csv")
#print(df)
#df.loc[df["ImageId"]]
#print(df.loc[df["ImageId"]])
val_louisiana =df.loc[df["ImageId"].isin(image_names)]
df = pd.read_csv("Germany_Training_Public/Germany_Training_Public_reference.csv")
val_germany = df.loc[df["ImageId"].isin(image_names)]
print(len(val_germany))
print(len(val_louisiana))
dfout = pd.concat([val_louisiana, val_germany])
dfout.to_csv("validation_reference.csv", index=False)
print(dfout)


