from flask import request, Flask, render_template, send_from_directory
import pandas as pd
import joblib
import numpy as np
from os import path

app = Flask(__name__)
app.config['upload_icons_folder']='icons'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/battle', methods=['POST','GET'])
def post():
    model = joblib.load('modeljoblib')

    pokemon1 = request.form['pokemon1']
    pokemon2 = request.form['pokemon2']

    pokemon = pd.read_csv('pokemon.csv', index_col=0)
    pokemon1_num = pokemon[pokemon.Name == pokemon1].index.values
    pokemon2_num = pokemon[pokemon.Name == pokemon2].index.values
    
    
    id1 = str(pokemon1_num).strip('[]')
    id2 = str(pokemon2_num).strip('[]')

    if pokemon1 == "":
        return render_template('notfound.html')
    else:
        if pokemon2 == "":
            return render_template('notfound.html')
        else:
            # za url
            address1 = 'icons/' + id1 + '.png'

            address2 = 'icons/' + id2 + '.png'

            if path.exists(address1) == True:
                icon1 = address1
            else:
                return render_template('notfound.html')

            if path.exists(address2) == True:
                icon2 = address2
            else:
                return render_template('notfound.html')

            pokemon_df = pd.DataFrame(list(zip(pokemon1_num, pokemon2_num)), columns = ["First_pokemon", "Second_pokemon"])
            
            def get_stats_dif(pokemons_df):
                
                pokemon_stats = ["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed","Generation","Legendary"] 
                stats_df = pokemon[pokemon_stats].T.to_dict("list")
                first_pokemon = pokemons_df.First_pokemon.map(stats_df)
                second_pokemon = pokemons_df.Second_pokemon.map(stats_df)
                new_list = []
                for i in range(len(first_pokemon)):
                   new_list.append(np.array(first_pokemon[i]) - np.array(second_pokemon[i]))
                new_pokemons_df = pd.DataFrame(new_list, columns = pokemon_stats)
                return new_pokemons_df
            
            new_pokemon_df = get_stats_dif(pokemon_df)
            prediction = model.predict(new_pokemon_df)

            if (prediction[0] == 1):
                winner = pokemon1
            elif (prediction[0] == 2):
                winner = pokemon2
            
            proba = model.predict_proba(new_pokemon_df)
            probamax = round(proba[0, prediction[0] - 1] * 100, 0)

            return render_template('pokemon.html', pokemon1 = pokemon1, pokemon2 = pokemon2, icon1 = icon1, icon2 = icon2, winner = winner, proba = probamax)

@app.route('/icons/<path:x>')
def picture(x):
    return send_from_directory('icons', x)

@app.errorhandler(404)
def notFound404(x):
    return render_template('notfound.html')

if __name__=='__main__':
    app.run(debug=True)
