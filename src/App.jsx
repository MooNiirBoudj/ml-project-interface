import { useState } from 'react';
import './App.css'; 
import lRimg1 from '/amazon/logistic-regression/image-matrix.png';
import lRimg2 from '/amazon/logistic-regression/image-roc.png';
import classificationData1 from '/amazon/logistic-regression/CorrelationMatrix.png';
import classificationData2 from '/amazon/logistic-regression/TargetDistribution.png'

// Data from the report
const teamMembers = [
  { name: "Boudjelal Mounir", email: "mounir.boudjelal@ensia.edu.dz", group: "S2G7", classification: "Logistic Regression", regression: "Linear Regression" },
  { name: "Belkebir Mustapha", email: "belkebir.mustapha@ensia.edu.dz", group: "S2G7", classification: "SVM", regression: "SVR" },
  { name: "Bennani Anes", email: "anes.bennani@ensia.edu.dz", group: "S2G7", classification: "Neural Networks", regression: "Neural Networks" },
  { name: "Mahmoudi Mohamed Eyad", email: "mohamed.eyad.mahmoudi@ensia.edu.dz", group: "S2G7", classification: "KNN & Naive Bayes", regression: "KNN" },
  { name: "Halimi Ilyes", email: "ilyes.halimi@ensia.edu.dz", group: "S2G5", classification: "Random Forests, LGBMBoost & XGBoost", regression: "Random Forests" },
];

const datasets = {
  classification: {
    name: "Amazon Employee Access Challenge",
    description: "This dataset from Kaggle contains historical data collected from 2010 & 2011. The objective is to predict whether an employee should be granted access based on historical data. The dataset contains 9 features in addition to the target feature, which is the ACTION attribute.",
    challenges: "Severe class imbalance (class 0 represents only about 5% of total instances)",
    size: "32,769 rows for training and 58,921 rows for testing",
    images: [
      classificationData2
,
      classificationData1
    ]
  },
  regression: {
    name: "Communities and Crime",
    description: "This dataset combines socio-economic data from the 1990 US Census, law enforcement data from the 1990 US LEMAS survey, and crime data from the 1995 FBI UCR.",
    challenges: "Contains missing values and requires feature engineering",
    size: "127 features and 1,994 instances",
    images: [
      "/api/placeholder/750/400",
      "/api/placeholder/750/350"
    ]
  }
};

const algorithms = {
  classification: [
    {
      name: "Logistic Regression",
      methodology: "Feature expansion through high-order categorical interactions was performed, followed by a greedy feature selection process to reduce the number of features from 2,008,898 to a select few that increase AUC. The final model was fine-tuned by optimizing the regularization strength parameter C.",
      results: "Mean AUC after 10-fold cross-validation: 0.90215",
      images: [
        lRimg1,
        lRimg2
      ]
    },
    {
      name: "Neural Networks",
      methodology: "Instead of one-hot encoding, embeddings were used to represent categorical features efficiently. The network architecture consisted of an input layer of size 384 (combined size of all embedded vectors), followed by hidden layers of sizes 192, 96, 48, and an output layer of size 1. Binary Cross-Entropy loss function and ReLU activation function were used.",
      results: "The embedding approach allowed for effective handling of high-cardinality categorical data.",
      images: [
        "/api/placeholder/550/400"
      ]
    },
    {
      name: "SVM",
      methodology: "Initially, a baseline SVM model with default parameters was tested, achieving 94% accuracy but failing to differentiate between classes due to imbalance. Hyperparameter tuning was conducted, optimizing for kernel type, regularization parameter C, and kernel coefficient gamma. The best configuration used an RBF kernel with C=100 and gamma=auto.",
      results: "While accuracy improved slightly to 94.13%, the ROC AUC was only 0.6430, indicating limited performance on the imbalanced dataset.",
      images: [
        "/api/placeholder/400/350",
        "/api/placeholder/400/350"
      ]
    },
    {
      name: "KNN",
      methodology: "K-Nearest Neighbors algorithm was implemented with optimized parameters.",
      results: "Accuracy: 0.9405, Precision: 0.9497, Recall: 0.9893, F1 Score: 0.9691, ROC AUC: 0.6898",
      images: [
        "/api/placeholder/550/400",
        "/api/placeholder/550/400"
      ]
    },
    {
      name: "Naive Bayes",
      methodology: "The Naive Bayes classifier was applied as a baseline probabilistic approach.",
      results: "Accuracy: 0.9135, Precision: 0.9446, Recall: 0.9649, F1 Score: 0.9547, ROC AUC: 0.5661",
      images: [
        "/api/placeholder/550/400",
        "/api/placeholder/550/400"
      ]
    },
    {
      name: "Random Forests",
      methodology: "To address the severe class imbalance, multiple resampling strategies were implemented alongside hyperparameter optimization. The Random Forest was tuned with 5-fold cross-validation, using 200 estimators, unlimited max depth, and minimum 5 samples per split. Four resampling approaches (SMOTE, ADASYN, Oversampling, Undersampling) were evaluated.",
      results: "SMOTE achieved the highest ROC AUC (0.84) with balanced performance. SMOTE provided the best balance between class separation and minority class detection.",
      images: [
        "/api/placeholder/550/400"
      ]
    },
    {
      name: "XGBoost vs LightGBM",
      methodology: "K-fold evaluation comparing LightGBM and XGBoost across multiple resampling techniques was performed. Various sampling methods including SMOTE, SMOTETomek, ADASYN, random oversampling, and undersampling were tested.",
      results: "LightGBM demonstrated superior robustness, maintaining strong test accuracy (82.7-90.3%) across all sampling methods. LightGBM with SMOTE was identified as the optimal solution, achieving 90.3% test accuracy and 0.905 AUC.",
      images: [
        "/api/placeholder/550/250"
      ]
    }
  ],
  regression: [
    {
      name: "Linear Regression",
      methodology: "Dimensionality reduction and feature selection were applied to improve model performance. PCA was used to select the top 14 principal components, capturing approximately 84% of total variance. A RANSAC regressor with a linear base estimator was applied to increase robustness against outliers.",
      results: "R² score of 0.6767 on training and 0.6544 on testing. MSE was 0.0176 for training and 0.0185 for testing. The closely aligned metrics between training and testing indicate that the model is both accurate and robust, with no signs of overfitting.",
      images: [
        "/api/placeholder/550/250",
        "/api/placeholder/550/250"
      ]
    },
    {
      name: "Neural Networks",
      methodology: "A fully-connected Neural Network was used with simple architecture given the tabular nature of the data. Three one-hidden-layer models were tested: one with hidden-size of 21 using ReLU, one with hidden-size of 62 using ReLU, and one with hidden-size of 21 using LeakyReLU. MSE loss function was used with the full dataset as a batch, 500 iterations, 0.01 learning rate, and Adam optimizer.",
      results: "The first model performed best with a 62% R² score (slightly under the model with 2 hidden layers at 63%), but was preferred for its simplicity and faster computation.",
      images: [
        "/api/placeholder/550/400"
      ]
    },
    {
      name: "SVR",
      methodology: "A baseline Support Vector Regression model was established, followed by hyperparameter tuning using greedy search over kernel type, C (regularization parameter), gamma (kernel coefficient), and epsilon (margin of tolerance). The optimal configuration used an RBF kernel, C=100, gamma=auto, and epsilon=0.1.",
      results: "The optimized model showed a 23.11% improvement in both MSE and R² score over the default model, demonstrating enhanced prediction accuracy and better fit to the data.",
      images: [
        "/api/placeholder/550/250",
        "/api/placeholder/550/250",
        "/api/placeholder/550/70",
        "/api/placeholder/550/250"
      ]
    },
    {
      name: "KNN",
      methodology: "An initial KNN regression model was trained with default parameters (n_neighbors=5). After applying PCA (retaining 98% variance), the feature space was reduced from 42 to 34 components. GridSearchCV was used to optimize hyperparameters: n_neighbors=10, weights='uniform', and p=2 (Euclidean distance).",
      results: "The baseline model achieved an MSE of 0.0207 and R² of 0.5684. After optimization with PCA, performance improved to an MSE of 0.0177/0.0187 and R² of 0.6819/0.6101 for training/validation sets respectively.",
      images: [
        "/api/placeholder/550/400"
      ]
    },
    {
      name: "Random Forests",
      methodology: "Feature selection was performed using a RandomForestRegressor with conservative hyperparameters (max_depth=5, min_samples_leaf=5). The final model implemented a regularized RandomForestRegressor with restricted tree complexity (max_depth=6, min_samples_split=20, min_samples_leaf=15) and stochastic subsampling (max_features=0.4, max_samples=0.7).",
      results: "The model showed strong generalization with tight train-test consistency (R² gap: 0.078, MSE ratio: 1.09), cross-validation stability (CV R²: 0.625 ± 0.012), and balanced performance (Test R²: 0.624, Test MSE: 0.018).",
      images: [
        "/api/placeholder/550/220"
      ]
    }
  ]
};

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null);
  
  const renderHome = () => (
    <div className="home-container">
      <div className="content-wrapper">
        <h1 className="title">Machine Learning Project</h1>
        <p className="subtitle">National Higher School Of Artificial Intelligence</p>
        
        <div className="card project-overview">
          <h2 className="card-title">Project Overview</h2>
          <p className="card-text">
            This machine learning project involved implementing various algorithms for both classification and regression tasks using two different datasets. Our team explored multiple approaches to solve real-world problems and compared their performance.
          </p>
          <div className="task-selection">
            <div 
              onClick={() => setActiveTab('classification')}
              className="task-card classification-card"
            >
              <h3 className="task-title">Classification Task</h3>
              <p className="task-dataset">Amazon Employee Access Challenge</p>
              <p className="task-description">Predict whether an employee should be granted access based on historical data</p>
            </div>
            <div 
              onClick={() => setActiveTab('regression')}
              className="task-card regression-card"
            >
              <h3 className="task-title">Regression Task</h3>
              <p className="task-dataset">Communities and Crime</p>
              <p className="task-description">Predict violent crime rates using socio-economic and law enforcement data</p>
            </div>
          </div>
        </div>
        
        <div className="card team-members">
          <h2 className="card-title">Team Members</h2>
          <div className="table-container">
            <table className="team-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Email</th>
                  <th>Group</th>
                  <th>Classification Task</th>
                  <th>Regression Task</th>
                </tr>
              </thead>
              <tbody>
                {teamMembers.map((member, index) => (
                  <tr key={index} className="table-row">
                    <td>{member.name}</td>
                    <td>{member.email}</td>
                    <td>{member.group}</td>
                    <td>{member.classification}</td>
                    <td>{member.regression}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDatasetInfo = (type) => {
    const dataset = datasets[type];
    const taskClass = type === 'classification' ? 'classification' : 'regression';
    
    return (
      <div className="card dataset-info">
        <h2 className={`card-title ${taskClass}-title`}>Dataset: {dataset.name}</h2>
        <p className="card-text">{dataset.description}</p>
        <div className="dataset-details">
          <div className={`dataset-detail ${taskClass}-detail`}>
            <h3 className="detail-title">Challenges</h3>
            <p>{dataset.challenges}</p>
          </div>
          <div className={`dataset-detail ${taskClass}-detail`}>
            <h3 className="detail-title">Dataset Size</h3>
            <p>{dataset.size}</p>
          </div>
        </div>
      </div>
    );
  };

  const renderDatasetImages = (type) => {
    const dataset = datasets[type];
    const taskClass = type === 'classification' ? 'classification' : 'regression';
    
    return (
      <div className="card dataset-visualizations">
        <h2 className={`card-title ${taskClass}-title`}>Dataset Visualizations</h2>
        <div className="dataset-images-container">
          {dataset.images.map((img, idx) => (
            <div key={idx} className="dataset-image-wrapper">
              <img 
                src={img} 
                alt={`${dataset.name} visualization ${idx+1}`} 
                className="dataset-image"
              />
              <p className="image-caption">
                {idx === 0 ? "Dataset Distribution" : "Feature Correlations"}
              </p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderTaskPage = (type) => {
    const taskAlgorithms = algorithms[type];
    const taskClass = type === 'classification' ? 'classification' : 'regression';
    
    return (
      <div className="task-page">
        <div className="content-wrapper">
          <button 
            onClick={() => {
              setActiveTab('home');
              setSelectedAlgorithm(null);
            }}
            className="back-button"
          >
            <span className="arrow">←</span> Back to Home
          </button>
          
          <h1 className={`page-title ${taskClass}-title`}>
            {type === 'classification' ? 'Classification Task' : 'Regression Task'}
          </h1>
          
          {renderDatasetInfo(type)}
          
          {/* Dataset images section - Added as requested */}
          {renderDatasetImages(type)}
          
          {!selectedAlgorithm ? (
            <>
              <h2 className={`section-title ${taskClass}-title`}>Select an Algorithm</h2>
              <div className="algorithms-grid">
                {taskAlgorithms.map((algo, index) => (
                  <div 
                    key={index}
                    className={`algorithm-card ${taskClass}-card`}
                    onClick={() => setSelectedAlgorithm(algo)}
                  >
                    <h3 className="algorithm-title">{algo.name}</h3>
                    <p className="click-hint">Click to view details</p>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="card algorithm-details">
              <div className="algorithm-header">
                <h2 className={`algorithm-detail-title ${taskClass}-title`}>{selectedAlgorithm.name}</h2>
                <button 
                  onClick={() => setSelectedAlgorithm(null)}
                  className="back-to-algorithms"
                >
                  Back to Algorithms
                </button>
              </div>
              
              <div className="detail-section">
                <h3 className={`section-subtitle ${taskClass}-subtitle`}>Methodology</h3>
                <p className="detail-text">{selectedAlgorithm.methodology}</p>
              </div>
              
              <div className="detail-section">
                <h3 className={`section-subtitle ${taskClass}-subtitle`}>Results</h3>
                <p className="detail-text">{selectedAlgorithm.results}</p>
              </div>
              
              <div className="detail-section">
                <h3 className={`section-subtitle ${taskClass}-subtitle`}>Visualizations</h3>
                <div className="visualizations-grid">
                  {selectedAlgorithm.images.map((img, idx) => (
                    <div key={idx} className="visualization-container">
                      <img src={img} alt={`${selectedAlgorithm.name} visualization ${idx+1}`} className="visualization-image" />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1 className="header-title">ML Project Dashboard</h1>
          <nav className="main-nav">
            <ul className="nav-list">
              <li>
                <button 
                  onClick={() => {
                    setActiveTab('home');
                    setSelectedAlgorithm(null);
                  }}
                  className={`nav-link ${activeTab === 'home' ? 'active' : ''}`}
                >
                  Home
                </button>
              </li>
              <li>
                <button 
                  onClick={() => {
                    setActiveTab('classification');
                    setSelectedAlgorithm(null);
                  }}
                  className={`nav-link ${activeTab === 'classification' ? 'active' : ''}`}
                >
                  Classification
                </button>
              </li>
              <li>
                <button 
                  onClick={() => {
                    setActiveTab('regression');
                    setSelectedAlgorithm(null);
                  }}
                  className={`nav-link ${activeTab === 'regression' ? 'active' : ''}`}
                >
                  Regression
                </button>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main className="main-content">
        {activeTab === 'home' && renderHome()}
        {activeTab === 'classification' && renderTaskPage('classification')}
        {activeTab === 'regression' && renderTaskPage('regression')}
      </main>
      
      <footer className="footer">
        <div className="footer-content">
          <p>© 2025 National Higher School Of Artificial Intelligence - Machine Learning Project - BM</p>
        </div>
      </footer>
    </div>
  );
}