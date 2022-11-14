#RootHair app

import flask
import sqlite3 as sql
from flask import request, jsonify, render_template
from functools import wraps
from flask import request, abort
import sys

app = flask.Flask(__name__)
app.config["DEBUG"] = True

#ROUTE0
@app.route('/')
def index():
   return render_template('homefinal.html')

con= sql.connect("TestDB.db")
cursor=con.cursor()


	
# Create some test data for our Recipe database in the form of a list of dictionaries.
#Recipe = [
#    {'id': 1,
#     'Dish': 'ButterChicken',
#     'Native': 'India'},
#     
#    {'id': 2,
#     'Dish': 'PadThai',
#     'Native': 'Singapore'}
#]
#
#
##ROUTE1: List all recipes
#@app.route('/Recipe/all', methods=['GET'])
#def api_all():
#    return jsonify(Recipe)
#
##ROUTE2: List recipe specified by id (1 or 2) entered via the URL
@app.route('/Roothair', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args.get('id'))
    else:
        return "Error: No id field provided. Please specify an id."
    
    
    try:
        con= sql.connect("TestDB.db")
        print('+=========================+')
        print('|  CONNECTED TO DATABASE  |')
        print('+=========================+')
    except Exception as e:
        sys.exit('error',e)
        
    cur=con.cursor()
    Recipe=cur.execute("""
            SELECT * FROM ROOTHAIR  
            WHERE ID=id
            """)
    
#    results=[]
#    # Loop through the data and match results that fit the requested ID.
#    # IDs are unique, but other fields might return many results
#    for recipe in Recipe:
#            results.append(recipe)
#
#    # Use the jsonify function from Flask to convert our list of
#    # Python dictionaries to the JSON format.
#    return jsonify(results)
#    con.close()
#
##ROUTE3: Add a route with API key verification for 'POST' method
#@app.route('/API', methods=['GET','POST'])
#
#def key_id():
#    if 'key' in request.args:
#        key = int(request.args.get('key'))
#        if key == 1234:
#            return render_template('recipe.html')
#        else:
#            return "Please provide the correct API key."
#    else:
#        return "Error: No key provided. Please specify your key."
#    
#
##ROUTE4: Post into database
#@app.route('/addrec',methods = ['POST', 'GET'])
#def addrec():
#   if request.method == 'POST':
#      
#      try:
#         
#         id = request.form["id"]
#         Dish = request.form["Dish"]
#         Native = request.form["Native"]
#         
#         
#         with sql.connect("CookingDatabase.db") as con:
#            cur = con.cursor()
#            cur.execute("INSERT INTO Recipes (id,Dish,Native) VALUES (?,?,?,?)",(id,Dish,Native))
#            
#            con.commit()
#            msg = "Record successfully added. Click on list on the homepage to view your entry!"
#
#      except:
#         con.rollback()
#         msg = "error in insert operation"
#      
#      finally:
#         return render_template("result.html",msg = msg)
#         con.close()
#
##ROUTE5
##return updated database in JSON format
#@app.route('/list')
#def list():
#   con = sql.connect("CookingDatabase.db")
#   con.row_factory = sql.Row
#   
#   cur = con.cursor()
#   result=cur.execute("select * from Recipes")
#   
#   results=[]
#   for row in result:
#       results.append({'id':row[0], 'Dish':row[1], 'Native':row[2]})
#  
#   
#   return jsonify(results)
      
if __name__ == '__main__':
   app.run(host="localhost", port=5000, debug=True)