import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={
        'Age': int (20),
        'BusinessTravel': 'Rarely',
        'DailyRate': int (250),
        'Department': 'Sales',
        'DistanceFromHome': int (12),
        'Education': 4,
        'EducationField': 'Other',
        'EnvironmentSatisfaction': int (3),
        'Gender': 'Male',
        'HourlyRate': int (50),
        'JobInvolvement': int (4),
        'JobLevel': int (3),
        'JobRole': 'Research Scientist',
        'JobSatisfaction': int (3),
        'MaritalStatus': 'Married',
        'MonthlyIncome': int (1244),
        'NumCompaniesWorked': int (4),
        'OverTime': 'Yes',
        'PerformanceRating': int (3),
        'RelationshipSatisfaction': int (2),
        'StockOptionLevel': 1,
        'TotalWorkingYears': int (20),
        'TrainingTimesLastYear': 3,
        'WorkLifeBalance': int (4),
        'YearsAtCompany': int (5),
        'YearsInCurrentRole': int (2),
        'YearsSinceLastPromotion': int (1),
        'YearsWithCurrManager': int (1)
})

print(r.json())