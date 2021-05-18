import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pokemon=pd.read_csv('pokemon.csv', index_col = 0)
combats=pd.read_csv('combats.csv')
test_data = pd.read_csv('tests.csv') 

combats_columns = ["First_pokemon","Second_pokemon","Winner"]
new_combats_data=combats[combats_columns].replace(pokemon.Name)

# ukoliko je prvi pokemon pobjednik stavi 1, ukoliko je drugi stavi 2    
combats.Winner[combats.Winner == combats.First_pokemon] = 1
combats.Winner[combats.Winner == combats.Second_pokemon] = 2

# funkcija koja na temelju primljenog dataframe koji se sastoji od dva pokemona
# zapisana u obliku rednog broja iz pokemon.csv traži njihove statse i računa 
# njihovu razliku, zapisuje u novi df te ga vraća
def get_stats_dif(pokemons_df):
    pokemon_stats = ["HP","Attack","Defense","Sp_Atk","Sp_Def","Speed","Generation","Legendary"] 
    stats_df = pokemon[pokemon_stats].T.to_dict("list")
    first_pokemon = pokemons_df.First_pokemon.map(stats_df)
    second_pokemon = pokemons_df.Second_pokemon.map(stats_df)
    new_list = []
    for i in range(len(first_pokemon)):
        new_list.append(np.array(first_pokemon[i]) - np.array(second_pokemon[i]))
    new_pokemon_df = pd.DataFrame(new_list, columns = pokemon_stats)
    return new_pokemon_df


# izdvajanje podataka za učenje
data = get_stats_dif(combats)
data = pd.concat([data, combats.Winner], axis = 1)

x_label = data.drop("Winner", axis = 1) 
y_label = data["Winner"] 

x_train, x_test, y_train, y_test = train_test_split(x_label, y_label, test_size = 0.3)

# testiranje različitih klasifikatora
import time
from sklearn.linear_model import LogisticRegression
start_time = time.time()
Log_Reg = LogisticRegression()
model_1 = Log_Reg.fit(x_train, y_train)
prediction = model_1.predict(x_test) 
print('Accuracy LR: ', accuracy_score(prediction, y_test) * 100)
print("Time passed: %s seconds " % (time.time() - start_time))

from sklearn.neighbors import KNeighborsClassifier
start_time = time.time()
KNN = KNeighborsClassifier(n_neighbors = 5)
model_2 = KNN.fit(x_train, y_train)
prediction = model_2.predict(x_test) 
print('Accuracy KNN: ', accuracy_score(prediction, y_test) * 100)
print("Time passed: %s seconds " % (time.time() - start_time))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
start_time = time.time()
Rand_Forest = RandomForestClassifier(n_estimators = 100, min_samples_split = 2, max_features = "auto",
                                     class_weight=("balanced_subsample"))
model_3 = Rand_Forest.fit(x_train, y_train) 
prediction = model_3.predict(x_test) 
print('Accuracy RFC: ', accuracy_score(prediction, y_test) * 100)
print("Time passed: %s seconds " % (time.time() - start_time))

start_time = time.time()
Grad_Boosting = GradientBoostingClassifier( n_estimators = 100, max_depth = 1, learning_rate = 1,
                                            random_state = 0)
model_4 = Grad_Boosting.fit(x_train, y_train) 
prediction = model_4.predict(x_test) 
print('Accuracy GB: ', accuracy_score(prediction, y_test) * 100)
print("Time passed: %s seconds " % (time.time() - start_time))

# vrlo sporo za pokretanje 80s+, Accuracy 90%
# from sklearn.svm import NuSVC
# start_time = time.time()
# Svc = NuSVC()
# model_5 = Svc.fit(x_train, y_train)
# prediction = model_5.predict(x_test) 
# print('Accuracy SVC: ', accuracy_score(prediction, y_test) * 100)
# print("Time passed: %s seconds " % (time.time() - start_time))

# najbolji accuracy ima random forest (acc ~ 95), a vrlo blizu njega je i gradioent boosting (acc ~ 94), koji je brži
# zbog bolje točnosti i ne prevlikog vremena izvođenja bira se random forest 

# testiranje na tests.csv danom na istoj stranici zajedno sa pokemon.csv i combats.cs
new_test_data = get_stats_dif(test_data)
prediction = model_3.predict(new_test_data)

test_data["Winner"] = [test_data["First_pokemon"][i] if prediction[i] == 1 else test_data["Second_pokemon"][i] for i in range(len(prediction))]
pokemon_test_names = test_data[combats_columns].replace(pokemon.Name)

print(pokemon_test_names[1:10])

import joblib
joblib.dump(model_3,'modeljoblib')