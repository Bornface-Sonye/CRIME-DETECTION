# Data sources
criminal_records = {
"John Doe": {"age": 35, "criminal_history": "None"},
"Jane Smith": {"age": 28, "criminal_history": "Previous theft"},
}
social_media_data = {
"John Doe": {"location": "New York", "education": "Bachelor's"},
"Jane Smith": {"location": "Chicago", "education": "Master's"},
}
# Merging data from criminal records and social media profiles
integrated_data = {}
for name in criminal_records:
     if name in social_media_data:
          integrated_data[name] = {**criminal_records[name], **social_media_data[name]}
# Print the integrated data
print(integrated_data)