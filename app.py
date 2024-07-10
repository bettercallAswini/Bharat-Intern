from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

# Route to fetch unique values for dropdown suggestions
@app.route('/suggestions/<field>', methods=['GET'])
def get_suggestions(field):
    if field == 'beds':
        suggestions = sorted(data['beds'].unique())
    elif field == 'baths':
        suggestions = sorted(data['baths'].unique())
    elif field == 'size':
        suggestions = sorted(data['size'].unique())
    elif field == 'zip_code':
        suggestions = sorted(data['zip_code'].unique())
    else:
        suggestions = []
    
    return jsonify({'suggestions': suggestions})

# Route to render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = int(request.form['beds'])
    bathrooms = int(request.form['baths'])
    size = float(request.form['size'])
    zipcode = int(request.form['zip_code'])

    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                              columns=['beds', 'baths', 'size', 'zip_code'])

    prediction = pipe.predict(input_data)[0]
    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True)
