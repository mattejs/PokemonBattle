from flask import request, Flask, render_template, send_from_directory
import pandas as pd
import joblib
import numpy as np
from os import path
import matplotlib.pyplot as plt
from flask_sqlalchemy import SQLAlchemy
import sqlite3

app = Flask(__name__)
app.config['upload_icons_folder']='icons'
app.config['upload_folder']='plots'
app.config['SQLALCHEMY_DATABASE_URI'] = 'DATABASE_URL'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///previous_battles.db'

db = SQLAlchemy(app)

class PreviousBattles(db.Model):
    id=db.Column('ID', db.Integer, primary_key=True)
    firstPokemon = db.Column(db.String(50), nullable=False)
    firstPokemonID = db.Column(db.Integer, nullable=False)
    firstImagePath = db.Column(db.String(50), nullable=False)
    secondPokemon = db.Column(db.String(50), nullable=False)
    secondPokemonID = db.Column(db.Integer, nullable=False)
    secondImagePath = db.Column(db.String(50), nullable=False)
    winner = db.Column(db.String(50), nullable=False)
    
    def __init__(self,firstPokemon,firstPokemonID,secondPokemon,secondPokemonID,winner, firstImagePath,secondImagePath):
    	self.firstPokemon = firstPokemon
    	self.firstPokemonID = firstPokemonID
    	self.secondPokemon = secondPokemon
    	self.secondPokemonID = secondPokemonID
    	self.winner = winner
    	self.firstImagePath = firstImagePath
    	self.secondImagePath = secondImagePath
    
    def __repr__(self):
    	return '<ID %r>' % self.id    
    
@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/list', methods=['GET', 'POST'])
def lista():
    conn = sqlite3.connect('previous_battles.db')
    cursor = conn.execute('SELECT * FROM previous_battles ORDER BY id DESC LIMIT 6')
    items = cursor.fetchall()
    return render_template('list.html', items=items)

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
                
                pokemon_stats = ["HP","Attack","Defense","Sp_Atk","Sp_Def","Speed","Generation","Legendary"] 
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
            
            hp1 = pokemon[pokemon.Name == pokemon1].HP.values[0]
            hp2 = pokemon[pokemon.Name == pokemon2].HP.values[0]
            att1 = pokemon[pokemon.Name == pokemon1].Attack.values[0]
            att2 = pokemon[pokemon.Name == pokemon2].Attack.values[0]
            def1 = pokemon[pokemon.Name == pokemon1].Defense.values[0]
            def2 = pokemon[pokemon.Name == pokemon2].Defense.values[0]
            spatk1 = pokemon[pokemon.Name == pokemon1].Sp_Atk.values[0]
            spatk2 = pokemon[pokemon.Name == pokemon2].Sp_Atk.values[0]
            spdef1 = pokemon[pokemon.Name == pokemon1].Sp_Def.values[0]
            spdef2 = pokemon[pokemon.Name == pokemon2].Sp_Def.values[0]
            speed1 = pokemon[pokemon.Name == pokemon1].Speed.values[0]
            speed2 = pokemon[pokemon.Name == pokemon2].Speed.values[0]
            
            legendary1 = pokemon[pokemon.Name == pokemon1].Legendary.values[0]
            legendary2 = pokemon[pokemon.Name == pokemon2].Legendary.values[0]
            
            if(legendary1 == 0):
                legendary1 = " - Not legendary -"
            else:
                legendary1 = " - Legendary -"
            
            if(legendary2 == 0):
                legendary2 = " - Not legendary -"
            else:
                legendary2 = " - Legendary -"
            
            generation1 = pokemon[pokemon.Name == pokemon1].Generation.values[0]
            generation2 = pokemon[pokemon.Name == pokemon2].Generation.values[0]
            
            avghp = pokemon[['HP']].mean().values[0]
            avgatt = pokemon[['Attack']].mean().values[0]
            avgdef = pokemon[['Defense']].mean().values[0]
            avgspatk = pokemon[['Sp_Atk']].mean().values[0]
            avgspdef = pokemon[['Sp_Def']].mean().values[0]
            avgspeed = pokemon[['Speed']].mean().values[0]
            
            max_hp = pokemon[['HP']].max().values[0]
            max_att = pokemon[['Attack']].max().values[0]
            max_def = pokemon[['Defense']].max().values[0]
            max_spatk = pokemon[['Sp_Atk']].max().values[0]
            max_spdef = pokemon[['Sp_Def']].max().values[0]
            max_speed = pokemon[['Speed']].max().values[0]
            
            min_hp = pokemon[['HP']].min().values[0]
            min_att = pokemon[['Attack']].min().values[0]
            min_def = pokemon[['Defense']].min().values[0]
            min_spatk = pokemon[['Sp_Atk']].min().values[0]
            min_spdef = pokemon[['Sp_Def']].min().values[0]
            min_speed = pokemon[['Speed']].min().values[0]
            
            plt.close()
            fug = plt.figure('Stats', figsize=(24,12))
            fug.suptitle("STATS COMPARISON", fontsize = 35)
            fug.tight_layout(0.2)
            
            
            plt.subplot(1, 6, 1)
            
            plt.figtext(0.01, 0.5 + 0.32, "Legendary status:", fontsize = 17)
            plt.figtext(0.01, 0.47 + 0.32, str(pokemon1) + ':', fontsize = 13)
            plt.figtext(0.012, 0.45 + 0.32 - 0.005, legendary1, fontsize = 14)
            plt.figtext(0.01, 0.42 + 0.32 - 0.005, str(pokemon2) + ':', fontsize = 13)
            plt.figtext(0.012, 0.40 + 0.32 - 0.01, legendary2, fontsize = 14)
            
            plt.figtext(0.01, 0.33 + 0.28 - 0.01, "Generations:", fontsize = 17)
            plt.figtext(0.01, 0.30 + 0.28 - 0.01, str(pokemon1) + ': ' + str(generation1), fontsize = 13)
            plt.figtext(0.01, 0.27 + 0.28 - 0.01, str(pokemon2) + ': ' + str(generation2), fontsize = 13)
            
            plt.figtext(0.01, 0.87 - 0.43 - 0.01, "Average stats:", fontsize = 17)
            plt.figtext(0.01, 0.83 - 0.43 - 0.01, "Avg HP: " + str(avghp), fontsize = 14)
            plt.figtext(0.01, 0.79 - 0.43 - 0.01, "Avg Att: " + str(avgatt), fontsize = 14)
            plt.figtext(0.01, 0.75 - 0.43 - 0.01, "Avg Def: " + str(avgdef), fontsize = 14)
            plt.figtext(0.01, 0.71 - 0.43 - 0.01, "Avg Sp. Atk: " + str(avgspatk), fontsize = 14)
            plt.figtext(0.01, 0.67 - 0.43 - 0.01, "Avg Sp. Def: " + str(avgspdef), fontsize = 14)
            plt.figtext(0.01, 0.63 - 0.43 - 0.01, "Avg Speed: " + str(avgspeed), fontsize = 14)
            
            plt.figtext(0.918, 0.77 + 0.05, "Min stats:", fontsize = 17)
            plt.figtext(0.918, 0.73 + 0.05, "Min HP: " + str(min_hp), fontsize = 14)
            plt.figtext(0.918, 0.69 + 0.05, "Min Att: " + str(min_att), fontsize = 14)
            plt.figtext(0.918, 0.65 + 0.05, "Min Def: " + str(min_def), fontsize = 14)
            plt.figtext(0.918, 0.61 + 0.05, "Min Sp. Atk: " + str(min_spatk), fontsize = 14)
            plt.figtext(0.918, 0.57 + 0.05, "Min Sp. Def: " + str(min_spdef), fontsize = 14)
            plt.figtext(0.918, 0.53 + 0.05, "Min Speed: " + str(min_speed), fontsize = 14)

            plt.figtext(0.918, 0.47 - 0.05, "Max stats:", fontsize = 17)
            plt.figtext(0.918, 0.43 - 0.05, "Max HP: " + str(max_hp), fontsize = 14)
            plt.figtext(0.918, 0.39 - 0.05, "Max Att: " + str(max_att), fontsize = 14)
            plt.figtext(0.918, 0.35 - 0.05, "Max Def: " + str(max_def), fontsize = 14)
            plt.figtext(0.918, 0.31 - 0.05, "Max Sp. Atk: " + str(max_spatk), fontsize = 14)
            plt.figtext(0.918, 0.27 - 0.05, "Max Sp. Def: " + str(max_spdef), fontsize = 14)
            plt.figtext(0.918, 0.23 - 0.05, "Max Speed: " + str(max_speed), fontsize = 14)
            
            plt.title('HP', fontsize = 20)
            plt.bar(pokemon1, hp1, width= 1, color = 'steelblue', edgecolor = "black")
            plt.xticks([])
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.yticks(fontsize = 12)
            plt.bar(pokemon2, hp2, width= 1, color = 'silver', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.legend([pokemon1, pokemon2], bbox_to_anchor=(-1.05,-0.13), loc = 'lower left', fontsize = 18)
            plt.xticks([])

            plt.subplot(1, 6, 2)
            plt.title('Attack', fontsize = 20)
            plt.bar(pokemon1, att1, width= 1, color = 'steelblue', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])
            plt.yticks(fontsize = 12)
            plt.bar(pokemon2, att2, width= 1, color = 'silver', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])

            plt.subplot(1, 6, 3)
            plt.title('Defense', fontsize = 20)
            plt.bar(pokemon1, def1, width= 1, color = 'steelblue', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])
            plt.yticks(fontsize = 12)
            plt.bar(pokemon2, def2, width= 1, color = 'silver', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])

            plt.subplot(1, 6, 4)
            plt.title('Special Attack', fontsize = 20)
            plt.bar(pokemon1, spatk1, width= 1, color = 'steelblue', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])
            plt.yticks(fontsize = 12)
            plt.bar(pokemon2, spatk2, width= 1, color = 'silver', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])

            plt.subplot(1, 6, 5)
            plt.title('Special Defense', fontsize = 20)
            plt.bar(pokemon1, spdef1, width= 1, color = 'steelblue', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])
            plt.yticks(fontsize = 12)
            plt.bar(pokemon2, spdef2, width= 1, color = 'silver', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])

            plt.subplot(1, 6, 6)
            plt.title('Speed', fontsize = 20)
            plt.bar(pokemon1, speed1, width= 1, color = 'steelblue', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])
            plt.yticks(fontsize = 12)
            plt.bar(pokemon2, speed2, width= 1, color = 'silver', edgecolor = "black")
            # plt.xticks(rotation = 20, fontsize = 15)
            plt.xticks([])
            

            address = './plots/' + pokemon1 + 'vs' + pokemon2 +'.png'
            urlgraph ='./graph/'+ pokemon1 +'vs'+ pokemon2 +'.png'
            plt.savefig(address)
            graph = urlgraph
            print(winner)
            dbData = PreviousBattles(firstPokemon = pokemon1,
            			      firstPokemonID = id1,
            			      secondPokemon = pokemon2,
            			      secondPokemonID = id2,
            			      winner = winner,
            			      firstImagePath = icon1,
            			      secondImagePath = icon2)
            
            try:
            	db.session.add(dbData)
            	db.session.commit()
            	
            	status = 'Successful'
            	
            except:
            	status = 'Error'
              
            return render_template('pokemon.html', pokemon1 = pokemon1, pokemon2 = pokemon2, icon1 = icon1, icon2 = icon2, winner = winner, proba = probamax, graph = graph, db_status='{}'.format(status))

@app.route('/icons/<path:x>')
def picture(x):
    return send_from_directory('icons', x)

@app.route('/graph/<path:x>')
def graph(x):
    return send_from_directory('plots', x)

@app.errorhandler(404)
def notFound404(x):
    return render_template('notfound.html')

if __name__=='__main__':
    db.create_all()
    app.run(debug=True)
