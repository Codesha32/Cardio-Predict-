
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")


class Theme:
    RED = "#1D2E1F"
    DARK = "#212529"
    WHITE = "#FFFFFF"
    SUCCESS = "#28A745"
    INFO = "#17A2B8"
    WARNING = "#FFC107"



def data_clean(df):
    """Fill missing values: mode for objects, median for numbers."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            if df[c].isnull().any():
                df[c].fillna(df[c].mode().iat[0] if not df[c].mode().empty else "Unknown", inplace=True)
        else:
            if df[c].isnull().any():
                df[c].fillna(df[c].median(), inplace=True)
    return df


def features(df, encoders=None):
    
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            mapped = []
            for val in df[col].astype(str):
                if val in le.classes_:
                    mapped.append(int(le.transform([val])[0]))
                else:
                    
                    mapped.append(len(le.classes_))
            df[col] = mapped
    return df, encoders


def balanced(X, y, random_state=42):
   
    np.random.seed(random_state)
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    Xs = []
    ys = []
    for cls in unique:
        X_cls = X[y == cls]
        n = len(X_cls)
        if n == 0:
            continue
        if n < max_count:
            idx = np.random.choice(np.arange(n), size=(max_count - n), replace=True)
            X_extra = X_cls[idx]
            X_cls = np.vstack([X_cls, X_extra])
        Xs.append(X_cls)
        ys.extend([cls] * X_cls.shape[0])
    X_bal = np.vstack(Xs)
    y_bal = np.array(ys)
    perm = np.random.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]



class HeartDisease:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=150, random_state=42)
        self.lr = LogisticRegression(max_iter=1000)
        self.scaler = RobustScaler()
        self.encoders = {}
        self.features = None
        self.trained = False

    def fit(self, df, target_col="Heart_Disease"):
       
        X = df.drop(columns=[target_col])
        y = df[target_col].values

       
        X = data_clean(X)
        X, self.encoders = features(X, encoders=None)
        self.features = X.columns.tolist()

       
        X_arr = X.values.astype(float)
        X_bal, y_bal = balanced(X_arr, y)
        X_scaled = self.scaler.fit_transform(X_bal)

       
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_bal, test_size=0.2, random_state=42, stratify=y_bal
        )

      
        print("[DEBUG] Training RandomForest...")
        self.rf.fit(X_train, y_train)
        print("[DEBUG] Training LogisticRegression...")
        self.lr.fit(X_train, y_train)

        
        y_rf = self.rf.predict(X_test)
        y_lr = self.lr.predict(X_test)
        try:
            y_rf_proba = self.rf.predict_proba(X_test)[:, 1]
            rf_auc = roc_auc_score(y_test, y_rf_proba)
        except Exception:
            rf_auc = None

        try:
            y_lr_proba = self.lr.predict_proba(X_test)[:, 1]
            lr_auc = roc_auc_score(y_test, y_lr_proba)
        except Exception:
            lr_auc = None

        print("[INFO] RF acc / auc:", accuracy_score(y_test, y_rf), rf_auc)
        print("[INFO] LR acc / auc:", accuracy_score(y_test, y_lr), lr_auc)

        self.trained = True
        self.performance = {
            "RandomForest": {
                "accuracy": accuracy_score(y_test, y_rf),
                "auc": rf_auc
            },
            "LogisticRegression": {
                "accuracy": accuracy_score(y_test, y_lr),
                "auc": lr_auc
            }
        }

    def _prepare_input(self, patient_dict):
        """Make a single-row DataFrame, clean, encode and scale."""
        df = pd.DataFrame([patient_dict])
        df = data_clean(df)
        df, _ = features(df, encoders=self.encoders)
    
        for col in self.features:
            if col not in df.columns:
                df[col] = 0
        df = df[self.features]
        return self.scaler.transform(df.values.astype(float))

    def predict(self, patient_dict):
        if not self.trained:
            return {"error": "model not trained"}

        X = self._prepare_input(patient_dict)
        out = {}

      
        try:
            p_rf = self.rf.predict_proba(X)[0][1]
        except Exception:
            p_rf = float(self.rf.predict(X)[0])  
        out['RandomForest'] = {"risk_score": p_rf, "prediction": "High Risk" if p_rf > 0.5 else "Low Risk"}

        
        try:
            p_lr = self.lr.predict_proba(X)[0][1]
        except Exception:
            p_lr = float(self.lr.predict(X)[0])
        out['LogisticRegression'] = {"risk_score": p_lr, "prediction": "High Risk" if p_lr > 0.5 else "Low Risk"}

        avg = (p_rf + p_lr) / 2.0
        level = "LOW"
        if avg > 0.7:
            level = "HIGH"
        elif avg > 0.4:
            level = "MODERATE"
        out['Ensemble'] = {"risk_score": avg, "prediction": ("High Risk" if avg > 0.5 else "Low Risk"), "level": level}
        return out


class HeartDiseaseApp:
    def __init__(self, root):
        self.root = root
        root.title("Heart Disease Prediction")
        root.geometry("1200x800")
        root.configure(bg=Theme.WHITE)

        self.model = HeartDisease()
        self.current_results = None

        self.build_ui()
       
        self.show_disclaimer()

        self.generate_synthetic_and_train()

    def build_ui(self):
        header = tk.Frame(self.root, bg=Theme.RED, height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="Cardio Predict ", font=("Arial", 24, "bold"),
                 fg="white", bg=Theme.RED).pack(side=tk.LEFT, padx=20, pady=12)
        tk.Label(header, text="Heart Disease Prediction", font=("Arial", 10), fg="#FFD6D6", bg=Theme.RED).pack(side=tk.LEFT)

        self.main = tk.Frame(self.root, bg=Theme.WHITE)
        self.main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.show_home()

    def show_disclaimer(self):
        msg = ("Disclaimer: Educational use only. Not medical advice.")
        messagebox.showinfo("Disclaimer", msg)

    def generate_synthetic_and_train(self):
        np.random.seed(0)
        n = 1500
        data = {
            "Age": np.random.randint(30, 80, n),
            "Gender": np.random.choice(["Male", "Female"], n),
            "Chest_Pain_Type": np.random.choice(["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"], n),
            "Resting_BP": np.random.randint(90, 180, n),
            "Cholesterol": 150 + np.random.gamma(2, 30, n),
            "Fasting_BS": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "Max_HR": 220 - np.random.randint(30, 80, n) + np.random.normal(0, 12, n),
            "Exercise_Angina": np.random.choice(["Yes", "No"], n),
            "Oldpeak": np.abs(np.random.normal(1, 1.2, n)),
            "ST_Slope": np.random.choice(["Up", "Flat", "Down"], n),
        }
        df = pd.DataFrame(data)

        score = (
            (df["Age"] > 55).astype(int) * 0.25 +
            (df["Cholesterol"] > 240).astype(int) * 0.2 +
            (df["Resting_BP"] > 140).astype(int) * 0.15 +
            (df["Exercise_Angina"] == "Yes").astype(int) * 0.25 +
            (df["Oldpeak"] > 2).astype(int) * 0.2 +
            np.random.normal(0, 0.25, len(df))
        )
        df["Heart_Disease"] = (score > 0.5).astype(int)

 
        print("[INFO] Training backend on synthetic data...")
        self.model.fit(df, target_col="Heart_Disease")
        self.show_home_status()

    def show_home_status(self):
        for widget in self.main.winfo_children():
            widget.destroy()
        tk.Label(self.main, text="Heart Disease Risk Assessment", font=("Arial", 18, "bold")).pack(pady=10)

        btn = tk.Button(self.main, text="Start New Assessment", font=("Arial", 12),
                        bg=Theme.SUCCESS, fg="white", command=self.show_assessment)
        btn.pack(pady=12)

        if self.model.trained:
            accuracies = [v['accuracy'] for v in self.model.performance.values() if v['accuracy'] is not None]
            best = max(accuracies) if accuracies else 0.0
            tk.Label(self.main, text=f"Models Ready | Accuracy ~ {best:.1%}", font=("Arial", 10), fg=Theme.INFO).pack()

    def show_home(self):
        self.show_home_status()

    def show_assessment(self):
        for w in self.main.winfo_children():
            w.destroy()

        tk.Label(self.main, text="Patient Information", font=("Arial", 16, "bold")).pack(pady=8)
        frame = tk.Frame(self.main)
        frame.pack(pady=6)

        self.entries = {}
        fields = [
            ("Age", "45"), ("Resting_BP", "120"), ("Cholesterol", "200"),
            ("Max_HR", "150"), ("Oldpeak", "1.0"),
            ("Gender", "Male", ["Male", "Female"]),
            ("Chest_Pain_Type", "Asymptomatic", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"]),
            ("Exercise_Angina", "No", ["Yes", "No"]),
            ("ST_Slope", "Flat", ["Up", "Flat", "Down"]),
            ("Fasting_BS", "0", ["0", "1"])
        ]

        for label, default, *opts in fields:
            row = tk.Frame(frame)
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text=label + ":", width=18, anchor="w").pack(side=tk.LEFT)
            if opts:
                var = tk.StringVar(value=default)
                widget = ttk.Combobox(row, textvariable=var, values=opts[0], state="readonly")
            else:
                var = tk.StringVar(value=default)
                widget = tk.Entry(row, textvariable=var)
            widget.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.entries[label] = var

        tk.Button(self.main, text="Predict Risk", font=("Arial", 12, "bold"),
                  bg=Theme.RED, fg="white", command=self.run_prediction).pack(pady=10)

    def run_prediction(self):
        try:
            patient = {}
            for k, var in self.entries.items():
                v = var.get()
                if k in ["Age", "Resting_BP", "Max_HR", "Fasting_BS"]:
                    patient[k] = int(str(v).split()[0])
                elif k in ["Cholesterol", "Oldpeak"]:
                    patient[k] = float(v)
                else:
                    patient[k] = v

            self.current_results = self.model.predict(patient)
            self.show_results()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def show_results(self):
        for w in self.main.winfo_children():
            w.destroy()
        res = self.current_results.get("Ensemble", {})
        risk = res.get("risk_score", 0.0)
        level = res.get("level", "LOW")
        color = Theme.RED if level == "HIGH" else Theme.WARNING if level == "MODERATE" else Theme.SUCCESS

        tk.Label(self.main, text="Heart Disease Risk Assessment", font=("Arial", 18, "bold")).pack(pady=8)
        tk.Label(self.main, text=f"RISK LEVEL: {level}", font=("Arial", 30, "bold"), fg=color).pack(pady=6)
        tk.Label(self.main, text=f"Probability: {risk:.1%}", font=("Arial", 12)).pack(pady=4)

        tk.Label(self.main, text="Recommendations", font=("Arial", 14, "bold")).pack(pady=8, anchor="w")
        tips = [
            "Consult a cardiologist immediately" if level == "HIGH" else "Do routine check-up",
            "Do heart-healthy diet (low salt, low cholesterol)",
            "Exercise 30+ minutes  daily ",
            "Monitor blood pressure and cholesterol",
            "Quit smoking and limit alcohol"
        ]
        for t in tips:
            tk.Label(self.main, text="• " + t, font=("Arial", 11)).pack(anchor="w", padx=30)

        tk.Button(self.main, text="New Assessment", font=("Arial", 11),
                  bg=Theme.INFO, fg="white", command=self.show_home).pack(pady=12)

# ---------------------------
# APP Run 
# ---------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseApp(root)
    root.mainloop()
