from flask import Flask, render_template, request
import pandas as pd
from io import StringIO
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

app = Flask(__name__)

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model from the same directory as the script
model_path = os.path.join(current_dir, 'RandomForest.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'csvFile' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['csvFile']

        # If the user does not select a file, submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # read csv file into dataframe
        user_df = pd.read_csv(file)

        user_df['Level'] = user_df['Level'].astype('category')
        user_df['Component'] = user_df['Component'].astype('category')

        user_df['Date'] = pd.to_datetime(user_df['Date'])
        user_df['Time'] = pd.to_datetime(user_df['Time'])

        user_df['Year'] = user_df['Date'].dt.year
        user_df['Month'] = user_df['Date'].dt.month
        user_df['Day'] = user_df['Date'].dt.day
        user_df['Hour'] = user_df['Time'].dt.hour
        user_df['Minute'] = user_df['Time'].dt.minute

        user_df = user_df.drop(['Date', 'Time'], axis=1)

        vectorizer = TfidfVectorizer()
        X_text_vectorized = vectorizer.fit_transform(user_df['Content'])

        # Convert non-numeric columns in X_other_features to sparse matrix
        X_other_features = csr_matrix(user_df[['Year', 'Month', 'Day', 'Hour', 'Minute']].values)

        # Combine the vectorized text data with other features
        X_combined = hstack([X_text_vectorized, X_other_features])

        # Make the prediction
        predictions = model.predict(X_combined)

        # Filter the DataFrame based on predictions
        incidents_df = user_df[predictions == 1]

        # Convert the DataFrame to HTML for rendering
        incidents_html = incidents_df.to_html(index=False)

        # Render the template with the prediction and incidents
        return render_template('result.html', incidents_html=incidents_html)

if __name__ == '__main__':
    app.run(debug=True)
