import pandas as pd
import numpy as np
import random

actions = ["Verify", "Check", "Validate", "Assess", "Ensure", "Test"]
features = ["login functionality", "payment processing", "data export", "session management",
            "profile update", "password reset", "search functionality", "user registration",
            "email notification", "file upload", "data synchronization", "report generation"]
areas = ["user dashboard", "admin panel", "checkout page", "settings page", "landing page",
         "help center", "transaction history", "user profile", "product listing", "order summary"]
conditions = ["user is logged in", "internet connection is slow", "user has admin privileges",
              "there are multiple sessions", "input data is invalid", "the system is under heavy load",
              "user is navigating from the homepage", "user inputs special characters"]
expected_outcomes = ["data is correctly saved", "user is redirected to the homepage", "an error message is displayed",
                     "the transaction is processed within 2 seconds", "the session is terminated after inactivity",
                     "user receives a confirmation email", "data is encrypted for security", "access is denied"]

# Function to generate more complex descriptions
def generate_complex_descriptions(n):
    descriptions = set()

    while len(descriptions) < n:
        action = random.choice(actions)
        feature = random.choice(features)
        area = random.choice(areas)
        condition = random.choice(conditions)
        expected_outcome = random.choice(expected_outcomes)
        description = f"{action} {feature} in {area} when {condition} to ensure {expected_outcome}"
        descriptions.add(description)

    return list(descriptions)

# Generate 700 unique, complex descriptions
descriptions = generate_descriptions(700)
complex_descriptions = generate_complex_descriptions(700)

df = pd.DataFrame({
    "Description": descriptions,
    "Test_ID": list(range(1, 701)),
    "Functionality_Test_Case": np.random.randint(2, size=700),
    "User_Interface_Test_Case": np.random.randint(2, size=700),
    "Performance_Test_Case": np.random.randint(2, size=700),
    "Integration_Test_Case": np.random.randint(2, size=700),
    "Usability_Test_Case": np.random.randint(2, size=700),
    "Database_Test_Case": np.random.randint(2, size=700),
    "Security_Test_Case": np.random.randint(2, size=700),
    "User_Acceptance_Test_Case": np.random.randint(2, size=700),
})

df["Description"] = complex_descriptions

# Save the updated DataFrame to a new Excel file
complex_file_path = "C/User/Test_Cases_V21.xlsx" 
#replace the path to your current path
df.to_excel(complex_file_path, index=False)

complex_file_path
