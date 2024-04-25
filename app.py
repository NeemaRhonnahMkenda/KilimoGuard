from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

df = pd.read_csv('datasets/MOCK_WEATHER_DATA .csv')

# Extract features and target variable
X = df.drop(['log_id', 'date', 'pest_infestation'], axis=1)
y = df['pest_infestation']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=42)

# Define categorical features
categorical_features = ['crop_type']

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42)),
])

# Model Training
pipeline.fit(X_train, y_train)


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/predictor')
def predictor():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        crop_type = request.form['crop_type']
        temperature = float(request.form['temperature'])
        date = pd.to_datetime(request.form['date'])
        rainfall = float(request.form['rainfall'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        soil_moisture = float(request.form['soil_moisture'])

        # Create a DataFrame from user input
        user_input = {
            'date': [date],
            'temperature': [temperature],
            'humidity': [humidity],
            'rainfall': [rainfall],
            'wind_speed': [wind_speed],
            'crop_type': [crop_type],
            'soil_moisture': [soil_moisture],
        }

        user_input_df = pd.DataFrame(user_input)

        # Convert 'date' to datetime
        user_input_df['date'] = pd.to_datetime(user_input_df['date'])

        # Ensure the 'crop_type' column is present in the user input DataFrame
        user_input_df['crop_type'] = user_input_df['crop_type'].astype('category')

        # Feature Extraction for User Input
        user_input_features = pipeline.named_steps['preprocessor'].transform(user_input_df)

        # Prediction for User Input
        user_probabilities = pipeline.named_steps['classifier'].predict_proba(user_input_features)

        # Assuming the second column of user_probabilities corresponds to the positive class (pest infestation)
        pest_likelihood = user_probabilities[0][1] * 100

        # Set the threshold for classification (e.g., 50%)
        threshold = 50

        # Check the likelihood and display HIGH or LOW
        if pest_likelihood > threshold:
            pest_infestation = "HIGH"
            # Get the entered crop type
            entered_crop_type = user_input_df['crop_type'].iloc[0]

            # Check if the entered crop type is in the pest_info dictionary
            pest_info = {
                'tomatoes': {
                    'pests': '\n 1. Aphids,\n 2. Whiteflies,\n 3. Hornworms \n',
                    'mitigation': {
                        'A: early_stage': 'Use insecticidal soap, plant companion crops like marigolds \n',
                        'B: mid_stage': 'Regularly inspect plants, use neem oil, introduce beneficial insects \n',
                        'C: late_stage': 'Harvest ripe tomatoes promptly, remove plant debris, practice crop rotation '
                                         '\n'
                    }
                },
                'maize': {
                    'pests': '\n 1. Armyworms, \n 2. Borers, \n 3. Cutworms \n \n',
                    'mitigation': {
                        'A: Early Stages': 'Practice clean cultivation, use biological control methods \n',
                        'B: Mid Stages': 'Implement crop rotation, use pheromone traps \n',
                        'C: Later Stages': 'Harvest maize promptly, destroy crop residues, use resistant varieties \n'
                    },
                    'Procedures': {
                        'Land Preparation': 'The piece of land for planting maize should be prepared early, before '
                                            'the onset of rains, for weeds to decompose before planting.',
                        '1': 'Plough the land and make it level with a fine tilth. Considering the size of the land, '
                             'machines like tractors or ox-drawn ploughs can be used, observing the correct spacing.',
                        '2': 'Mix soil with manure and biochar for efficient and improved nutrient uptake as well as '
                             'stabilizing soil pH.',
                        '3': 'Make holes at a spacing of 90 x 30–50 cm if soil fertility is low or 75 x 25–50 cm if '
                             'soil fertility is relatively high.',
                        '4': 'Place 1 or 2 seeds per hole, or alternate 1 and 2 seeds at a depth of about 4 cm if the '
                             'soil is moist and about 10 cm if the soil is dry.',
                        '5': 'Cover the seeds with loose soil.'
                    },
                    'Operations': {
                        'Thinning and Gapping': 'Gapping is done to replace seeds that did not germinate after others '
                                                'germinated completely. Thinning is done when maize has grown to '
                                                'about 15 cm in height by removing weak and deformed seedlings to '
                                                'make space for healthy seedlings in a hole.',
                        'Fertilizer application': 'To achieve maximum yield, fertilizer should be applied on time. '
                                                  'Manure and biochar can also be added to soil with little or no '
                                                  'organic matter. When planting manually, thoroughly mix soil with a '
                                                  'teaspoonful of fertilizer into each planting hole to ensure that '
                                                  'it doesn’t burn the seeds. Place the seeds on top of the soil and '
                                                  'feel for softness. DAP is recommended for planting because it '
                                                  'contains phosphorous, which helps in root development.',
                        'Top dressing': 'Maize can be top dressed with CA 2-3 weeks after planting or when it is 45 '
                                        'cm (1 ft) high. One teaspoon of fertilizer should be applied to the base of '
                                        'each plant, 15 cm away from the plant in a ring or along the row. Top dress '
                                        'in two stages in areas with heavy rainfall: the first six weeks after sowing '
                                        'and the second 10-15 days later, or just before tussling. In areas '
                                        'experiencing low rainfall, topdressing is done only once at a rate of 50–100 '
                                        'kg per acre. Using CAN and urea for topdressing is good because it fixes '
                                        'nitrogen in the soil. Nitrogen increases the green color of the leaves to '
                                        'make food for the plant.',
                        'Weeding': 'Remove weeds mechanically, manually, or by using herbicides to prevent them from '
                                   'competing with the crops for nutrients, water, and light. First weeding, '
                                   'if done manually, should be done three weeks after planting, depending on the '
                                   'intensity of weeds in the field. Herbicides can be applied in two phases: '
                                   'pre-emergence, which is used before the maize germinates and weeds appear, '
                                   'and post-emergence, which is applied after the maize and weeds germinate.',
                        'Harvesting': 'Normally, each maize stalk should yield one large ear of maize, but in ideal '
                                      'conditions, the stalk can yield a second, slightly smaller ear that matures '
                                      'slightly later than the first. Maize is ready for harvesting when the kernels '
                                      'within the husks are well packed and produce a milky substance when the kernel '
                                      'is punctured.'
                    }
                },
                'potatoes': {
                    'pests': '\n 1. Colorado Potato Beetles,\n 2. Aphids,\n 3. Flea Beetles \n',
                    'mitigation': {
                        'A: early_stage': 'Remove and destroy infested leaves, use insecticidal soap \n',
                        'B: mid_stage': 'Rotate crops, plant potatoes away from tomatoes, peppers \n',
                        'C: late_stage': 'Harvest potatoes when mature, remove plant debris, practice crop rotation \n'
                    }
                },
                'beans': {
                    'pests': '\n 1. Aphids, \n 2. Mexican Bean Beetles, \n 3. Thrips \n',
                    'mitigation': {
                        'A: early_stage': 'Use insecticidal soap, encourage natural predators \n',
                        'B: mid_stage': 'Introduce ladybugs, handpick beetles, rotate crops \n',
                        'C: late_stage': 'Harvest beans regularly, remove debris, practice crop rotation \n'
                    }
                },
                'wheat': {
                    'pests': '\n 1. Aphids, \n 2. Hessian Fly, \n 3. Armyworms \n',
                    'mitigation': {
                        'A: early_stage': 'Select resistant wheat varieties, monitor for aphids \n',
                        'B: mid_stage': 'Practice crop rotation, use insecticidal soap sparingly \n',
                        'C: late_stage': 'Harvest wheat promptly, destroy crop residues, practice clean cultivation \n'
                    }
                }
            }

            pest_info = pest_info.get(entered_crop_type, {})
        else:
            pest_infestation = "LOW"
            pest_info = {}

            # Calculate precision
        y_pred = (user_probabilities[:, 1] > threshold).astype(int)  # Convert probabilities to binary predictions
        precision = precision_score(y_true=[1], y_pred=y_pred)  # Compute precision
        print("Precision:", precision)

        return render_template('result.html', pest_likelihood=pest_likelihood, pest_infestation=pest_infestation,
                               pest_info=pest_info, user_input_df=user_input_df)


@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')


@app.route('/waitlist')
def waitlist():
    return render_template('waitlist.html')


if __name__ == '__main__':
    app.run()
