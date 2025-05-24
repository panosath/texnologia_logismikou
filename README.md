# Molecular Biology Data Analysis Web Application

This interactive web application provides tools for data analysis and machine learning on molecular biology data. Built with Streamlit, it integrates various analysis modules into a user-friendly interface.

## Features

The application includes multiple modules:

1. **Data Upload**: Import CSV files for analysis
2. **K-Means Clustering**: Basic clustering of your data
3. **Adjustable Clustering**: K-Means with customizable parameters
4. **Interactive Plotting**: Create custom scatter plots
5. **Multi-tab Interface**: Explore data through different views
6. **Sidebar Settings**: Configure your analysis with a sidebar
7. **Dynamic Filtering**: Filter your data based on text input
8. **Train Classification Model**: Build a Random Forest classifier
9. **Prediction**: Use trained models to make predictions
10. **About the Project Team**: Information about the creators of this web app

## Running the Application

### Using Docker (Recommended)

1. Make sure Docker and Docker Compose are installed on your system and Docker Engine is running
2. Clone this repository or download as .zip and unzip to a folder
3. Navigate to the project directory and open Command Line (type "cmd" at the address bar, if on Windows)
4. First run the following command: "Docker build -t bioapp ."
5. Afterwards, run the following command: "Docker run -p 8501:8501 bioapp"
