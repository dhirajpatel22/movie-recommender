# Movie Recommender System

A machine learning-powered movie recommendation web application built with **Python**, **Flask**, and **scikit-learn**. The system uses **collaborative filtering** with a **SVD** model trained on The Movies Dataset to recommend movies similar to a user's favorite film. A comparison between the a **KNN** and **SVD** models is done to justify
choosing **SVD** for the web app. 

## Features

- Search for a movie title with fuzzy matching support
- Receive personalized movie recommendations
- Fast recommendations using preprocessed data structures
- Simple and responsive Flask web interface
- Built using real-world MovieLens ratings data
- Collaborative filtering powered by KNN

## Tech Stack

| Category | Technology |
|-----------|------------|
| Backend | Flask |
| Machine Learning | scikit-learn (SVD) |
| Data Processing | Pandas, SciPy |
| Frontend | HTML, CSS|
| Dataset | MovieLens |

## Project Structure

```text
movie-recommender/
├── data/                  # Movie data CSVs from Kaggle The Movies Dataset
├── requirements.txt       # Python dependencies
├── model.ipynb            # Data preprocesseing & model training
├── README.md
├── flask_app/ 
    ├── app.py             # Flask application
    ├── recommend.py       # Recommendation logic
    ├── templates/
        ├── index.html
        ├── movie.html
        └── try-again.html
    ├── static/
        └── style.css
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/dhirajpatel22/movie-recommender.git
cd movie-recommender
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

### Step 1: Generate the Model Files

Open and run model.ipynb from start to finish. The notebook trains the recommendation model and generates the required serialized files (the model and recommendation data).
These files are saved as `.pkl` files.

### Step 2: Move Generated Files

Move `svd_model.pkl` and `recommendation_data.pkl` files into the `flask_app/` directory.

Example:

```text
movie-recommender/
├── data/                 
├── requirements.txt       
├── model.ipynb           
├── README.md
├── flask_app/ 
    ├── app.py            
    ├── recommend.py             
    ├── svd_model.pkl            # model file
    ├── recommendation_data.pkl  # data file
    ├── templates/
        ├── index.html
        ├── movie.html
        └── try-again.html
    ├── static/
        └── style.css
```

### Step 3: Start the Flask Server

```bash
cd flask_app
python app.py
```

### Step 4: Open the Application

Navigate to:

```text
http://127.0.0.1:5000
```

## How It Works

1. The user enters a movie title.
2. The system finds the closest matching movie using fuzzy string matching.
3. A SVD collaborative filtering model identifies movies with similar user-rating patterns.
4. The top recommendations are returned and displayed on the webpage.

## Performance Optimization

To improve response times, the application preprocesses the movie dataset and stores key data structures as serialized files. Instead of rebuilding the user-movie matrix every time the application starts, the recommender loads:

- Sparse user-movie rating matrix
- Movie index mappings
- Movie lookup tables

This significantly reduces startup time and allows recommendations to be generated much faster.

## Dataset

This project uses The Movies Dataset from kaggle. https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/?select=movies_metadata.csv

## Future Improvements

- Add content-based filtering using genres and movie metadata
- Implement hybrid recommendation models
- Deploy the application to Render or Railway
- Integrate movie posters and metadata using an external API
- Add user accounts and personalized recommendations
- Explore matrix factorization techniques such as SVD
- Experiment with neural recommendation systems

## Author

**Dhiraj Patel**

- Boston University — B.S. in Data Science
- GitHub: https://github.com/dhirajpatel22

## License

This project is licensed under the MIT License.

---

If you found this project interesting, consider giving the repository a star on GitHub.