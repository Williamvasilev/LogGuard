from flask import Flask, render_template, request, redirect, url_for, g
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

DATABASE = 'users.db'

# Connect to SQLite database
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

# Create a table to store user credentials
def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY,
                            username TEXT UNIQUE NOT NULL,
                            password TEXT NOT NULL
                        )''')
        db.commit()
        
# Initialize the database
init_db()

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model from the same directory as the script
model_path = os.path.join(current_dir, 'RandomForest.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/homepage.html')
def homepage():
    return render_template('homepage.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/predict.html')
def predict_page():
    return render_template('predict.html')

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

        # Group incidents by their content and convert to dictionary
        grouped_incidents = incidents_df.groupby('Content').apply(lambda x: x.to_html(index=False)).to_dict()

        # Calculate number of incidents
        num_incidents = len(incidents_df)
        num_logs = len(user_df)

        # Render the template with the grouped incidents
        return render_template('result.html', grouped_incidents=grouped_incidents, num_incidents=num_incidents, num_logs=num_logs)

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('index.html')

# Sign-in and Sign-up form submission routes
@app.route('/signin', methods=['POST'])
def signin_post():
    username = request.form.get('username')
    password = request.form.get('password')

    # Check if the username exists in the database
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()

    if user and check_password_hash(user[2], password):
        # Successful sign-in, redirect to homepage
        return redirect(url_for('homepage'))
    else:
        # If username or password is incorrect, redirect back to sign-in page with an error message
        return render_template('signin.html', error='Invalid username or password.')

@app.route('/signup', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    password = request.form.get('password')

    # Hash the password before storing it
    hashed_password = generate_password_hash(password)

    # Check if the username already exists in the database
    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        return render_template('index.html', error='Username already exists.')
    else:
        # Insert the new user into the database
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        db.commit()
        # Redirect to the sign-in page
        return redirect(url_for('signin'))

if __name__ == '__main__':
    app.run(debug=True)
