import gradio as gr
import joblib
import pandas as pd
import numpy as np

# 1. Load the model
# Ensure the .pkl file is in the same folder
try:
    model = joblib.load('best_gym_crowd_model.pkl')
    print("Model loaded successfully.")
except:
    print("Error: Model not found. Ensure you have 'best_gym_crowd_model.pkl'.")
    # Create a dummy model so the interface displays even without the file (for testing)
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor()
    # Fit on dummy data just for initialization
    model.fit(np.zeros((1, 16)), [0])


# 2. Prediction function (Real-time Feature Engineering)
def predict_crowd(hour, month_str, day_str, temperature, semester_status, is_holiday):
    # --- A. Convert user inputs to numbers ---

    # Mapping days
    days_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    day_of_week = days_map[day_str]

    # Mapping months
    months_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    month = months_map[month_str]

    # Weekend logic
    is_weekend = 1 if day_of_week >= 5 else 0

    # Holiday logic
    is_holiday_bin = 1 if is_holiday else 0

    # Semester logic
    if semester_status == "Start of Semester":
        is_start = 1
        is_during = 1
    elif semester_status == "During Semester":
        is_start = 0
        is_during = 1
    else:  # Semester Break
        is_start = 0
        is_during = 0

    # --- B. Feature Engineering (Recalculate complex features) ---

    # Cyclical Features (Math)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Interaction Features
    weekend_hour = is_weekend * hour
    semester_temp = is_during * temperature

    # --- C. Create final DataFrame ---
    # The order of columns MUST be exactly the same as during training
    features = pd.DataFrame([{
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday_bin,
        'temperature': temperature,
        'is_start_of_semester': is_start,
        'is_during_semester': is_during,
        'month': month,
        'hour': hour,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'weekend_hour': weekend_hour,
        'semester_temp': semester_temp
    }])

    # Prediction
    prediction = model.predict(features)[0]
    result = int(max(0, prediction))  # No negative numbers

    # Contextual message
    if result < 20:
        msg = "🟢 Empty (Great time to go!)"
    elif result < 60:
        msg = "🟡 Moderate (Normal crowd)"
    else:
        msg = "🔴 Crowded (Maybe wait a bit?)"

    return f"{result} People", msg


# 3. Create Gradio interface
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
)

with gr.Blocks(theme=theme, title="Gym Crowd AI") as demo:
    gr.Markdown(
        """
        # 🏋️ Campus Gym Crowd Predictor
        **Plan your workout smarter.** This AI predicts how many people are at the gym based on time, weather, and semester schedule.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📅 Date & Time")
            hour_input = gr.Slider(0, 23, step=1, label="Hour of Day (24h)", value=17)
            day_input = gr.Dropdown(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                label="Day of Week", value="Wednesday"
            )
            month_input = gr.Dropdown(
                ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"],
                label="Month", value="September"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🌤️ Context")
            temp_input = gr.Slider(30, 100, label="Temperature (°F)", value=70)
            semester_input = gr.Radio(
                ["Start of Semester", "During Semester", "Semester Break"],
                label="Semester Status", value="During Semester"
            )
            holiday_input = gr.Checkbox(label="Is it a Holiday?", value=False)

            predict_btn = gr.Button("🔮 Predict Crowdedness", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Prediction Results")
            output_number = gr.Label(label="Estimated People Count")
            output_msg = gr.Textbox(label="Status", interactive=False)

    # Link the button to the function
    predict_btn.click(
        fn=predict_crowd,
        inputs=[hour_input, month_input, day_input, temp_input, semester_input, holiday_input],
        outputs=[output_number, output_msg]
    )

    gr.Markdown("---")
    gr.Markdown("*Model trained with Random Forest/Gradient Boosting on historical campus data.*")

# Launch the app
if __name__ == "__main__":
    demo.launch()