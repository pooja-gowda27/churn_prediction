import streamlit as st
import pandas as pd
import joblib
import json, os, hashlib, re, datetime, uuid
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Churn Admin Panel", layout="wide")
st.markdown(
    "<div style='position:fixed;top:8px;left:8px;z-index:9999;color:#111;opacity:0.55;font-size:12px'>App running</div>",
    unsafe_allow_html=True,
)

USERS_FILE_PATH = "auth_users.json"
PRED_HISTORY_PATH = "prediction_history.json"
AUDIT_LOG_PATH = "audit_log.json"

# -----------------------
# Utils
# -----------------------
def _hash_password(pw): return hashlib.sha256(pw.encode()).hexdigest()

def _load_users():
    if not os.path.exists(USERS_FILE_PATH): return {"users": []}
    with open(USERS_FILE_PATH,"r") as f: return json.load(f)
def _save_users(users): 
    with open(USERS_FILE_PATH,"w") as f: json.dump(users,f,indent=2)

def _email_is_valid(email):
    return re.match(r"^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$", email) is not None

def _password_is_strong(pw):
    if len(pw)<8: return False,"Min 8 chars"
    if not re.search(r"[A-Z]",pw): return False,"Include uppercase"
    if not re.search(r"[a-z]",pw): return False,"Include lowercase"
    if not re.search(r"[0-9]",pw): return False,"Include digit"
    if not re.search(r"[!@#$%^&*()_+\-=[\]{};':\",.<>/?]",pw): return False,"Include special char"
    return True,""

def _find_user_by_email(email):
    for u in _load_users().get("users",[]): 
        if u.get("email","").lower()==email.lower(): return u
    return None

def _register_admin(email,pw):
    if not _email_is_valid(email): return False,"Invalid email"
    strong,msg=_password_is_strong(pw)
    if not strong: return False,msg
    data=_load_users()
    if _find_user_by_email(email): return False,"User exists"
    data.setdefault("users",[]).append({"email":email,"password_hash":_hash_password(pw),"role":"admin"})
    _save_users(data)
    return True,"Registered"

def _authenticate_admin(email,pw):
    u=_find_user_by_email(email)
    if not u or u.get("role")!="admin": return False
    return u.get("password_hash")==_hash_password(pw)

def _reset_password(email,new_pw):
    if not _email_is_valid(email): return False,"Invalid email"
    strong,msg=_password_is_strong(new_pw)
    if not strong: return False,msg
    users=_load_users()
    for u in users.get("users",[]):
        if u.get("email","").lower()==email.lower():
            u["password_hash"]=_hash_password(new_pw)
            _save_users(users)
            return True,"Password reset"
    return False,"User not found"

def _log_action(action,email):
    logs=[]
    if os.path.exists(AUDIT_LOG_PATH):
        try: logs=json.load(open(AUDIT_LOG_PATH))
        except: logs=[]
    logs.append({"timestamp":datetime.datetime.now().isoformat(),"email":email,"action":action})
    with open(AUDIT_LOG_PATH,"w") as f: json.dump(logs,f,indent=2)

def _save_prediction_history(entry: dict):
    history = _load_prediction_history()
    history.append(entry)
    try:
        with open(PRED_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving prediction history: {e}")

def _load_prediction_history() -> list:
    """Loads prediction history safely, resets if file is corrupted."""
    if not os.path.exists(PRED_HISTORY_PATH):
        return []

    try:
        with open(PRED_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # Corrupted file -> reset to empty list
        st.warning("Prediction history file was corrupted. Resetting it.")
        with open(PRED_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []


def _load_model_and_columns():
    try: 
        model=joblib.load("churn_model.pkl")
        cols=list(joblib.load("model_columns.pkl"))
        return model,cols
    except Exception as e: 
        st.error(f"Load model error: {e}")
        return None,[]

# -----------------------
# Styles
# -----------------------
def inject_styles():
    st.markdown("""
    <style>
    .stApp {background:linear-gradient(135deg,#8EC5FC,#E0C3FC);}
    .card {background:#fff; border-radius:16px; padding:20px; box-shadow:0 14px 30px rgba(0,0,0,0.12); margin-bottom:20px;}
    .btn-gradient {background:linear-gradient(90deg,#7b2ff7,#f107a3); color:white; border-radius:999px; font-weight:700;}
    .badge-churn {background:linear-gradient(90deg,#ff416c,#ff4b2b); padding:10px 14px; border-radius:12px; color:#fff; font-weight:800;}
    .badge-nochurn {background:linear-gradient(90deg,#00b09b,#96c93d); padding:10px 14px; border-radius:12px; color:#fff; font-weight:800;}
    </style>
    """,unsafe_allow_html=True)

# -----------------------
# Sidebar Navigation
# -----------------------
def sidebar_menu():
    st.sidebar.title("Admin Panel")
    st.sidebar.markdown(f"👤 Logged in as: {st.session_state['auth']['email']}")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard",
            "Prediction History",
            "User Management",
            "Model Performance",
            "Data Upload",
            "Data Analysis",  # ✅ Added here
            "Audit Log",
            "Settings & Help",
            "Logout"
        ]
    )
    return page

# -----------------------
# Pages
# -----------------------
def page_dashboard():
    st.markdown("<div class='card'><h2>📊 Customer Churn Prediction</h2></div>",unsafe_allow_html=True)
    model,model_cols=_load_model_and_columns()
    if not model: return
    with st.form("single_pred"):
        Age=st.slider("Age",18,90,30)
        Tenure=st.number_input("Tenure (Months)",0,72,12)
        Referrals=st.number_input("Number of Referrals",0,50,0)
        Monthly=st.number_input("Monthly Charge",0.0,500.0,50.0)
        Total=st.number_input("Total Charges",0.0,10000.0,100.0)
        Gender=st.selectbox("Gender",["Female","Male"])
        Married=st.selectbox("Married",["No","Yes"])
        submit=st.form_submit_button("Predict",help="Predict churn for this customer")
    if submit:
        data={"Age":Age,"Tenure_in_Months":Tenure,"Number_of_Referrals":Referrals,
              "Monthly_Charge":Monthly,"Total_Charges":Total,"Gender":Gender,"Married":Married}
        df=pd.get_dummies(pd.DataFrame([data]))
        df=df.reindex(columns=model_cols,fill_value=0)
        pred=model.predict(df)[0]
        label="Churn" if pred==1 else "No Churn"
        badge_class="badge-churn" if pred==1 else "badge-nochurn"
        st.markdown(f"<div class='{badge_class}'>Prediction: {label}</div>",unsafe_allow_html=True)
        _log_action("Single prediction",st.session_state['auth']['email'])
        _save_prediction_history({**data,"prediction":label,"timestamp":datetime.datetime.now().isoformat()})

def page_history():
    st.markdown("<div class='card'><h2>📜 Prediction History</h2></div>", unsafe_allow_html=True)
    history = _load_prediction_history()
    if not history:
        st.info("No predictions yet.")
        return
    try:
        df = pd.DataFrame(history)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "history.csv")
    except Exception as e:
        st.error(f"Error loading prediction history: {e}")
        st.warning("Prediction history file may be corrupted. Try clearing history.")
    if st.button("Clear History"):
        if os.path.exists(PRED_HISTORY_PATH):
            os.remove(PRED_HISTORY_PATH)
            st.success("Prediction history cleared. Please reload the page.")

def page_user_mgmt():
    st.markdown("<div class='card'><h2>👥 User Management</h2></div>",unsafe_allow_html=True)
    users=_load_users().get("users",[])
    df=pd.DataFrame(users)
    st.dataframe(df)
    with st.form("add_user"):
        st.subheader("Add New Admin User")
        email=st.text_input("Email")
        pw=st.text_input("Password",type="password")
        role=st.selectbox("Role",["admin"])
        submit=st.form_submit_button("Add User")
        if submit:
            success,msg=_register_admin(email,pw)
            if success: st.success(msg)
            else: st.error(msg)

def page_model_perf():
    st.markdown("<div class='card'><h2>📈 Model Performance</h2></div>",unsafe_allow_html=True)
    st.write("Placeholder metrics")
    st.write("Accuracy: 0.92")
    # Dummy confusion matrix
    y_true=[0,1,0,1,0,1,0,0,1,1]
    y_pred=[0,1,0,1,0,0,0,0,1,1]
    cm=confusion_matrix(y_true,y_pred)
    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
    st.pyplot(fig)
    # Dummy ROC curve
    fpr,tpr,_=roc_curve(y_true,y_pred)
    roc_auc=auc(fpr,tpr)
    st.write(f"AUC: {roc_auc:.2f}")
    fig2,ax2=plt.subplots()
    ax2.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
    ax2.plot([0,1],[0,1],"--",color="gray")
    ax2.set_xlabel("FPR");ax2.set_ylabel("TPR");ax2.set_title("ROC Curve");ax2.legend()
    st.pyplot(fig2)

def page_batch_upload():
    st.markdown("<div class='card'><h2>📤 Batch Prediction</h2></div>",unsafe_allow_html=True)
    uploaded=st.file_uploader("Upload CSV",type=["csv"])
    model,model_cols=_load_model_and_columns()
    if uploaded:
        df=pd.read_csv(uploaded)
        st.dataframe(df)
        if st.button("Run Batch Prediction"):
            df_dummies=pd.get_dummies(df)
            df_dummies=df_dummies.reindex(columns=model_cols,fill_value=0)
            preds=model.predict(df_dummies)
            df["Prediction"]=["Churn" if p==1 else "No Churn" for p in preds]
            st.dataframe(df)
            for i,row in df.iterrows():
                _save_prediction_history({**row.to_dict(),"timestamp":datetime.datetime.now().isoformat()})
            _log_action("Batch prediction",st.session_state['auth']['email'])
            st.success("Batch predictions saved!")

def page_audit_log():
    st.markdown("<div class='card'><h2>📝 Audit Log</h2></div>",unsafe_allow_html=True)
    logs=[]
    if os.path.exists(AUDIT_LOG_PATH):
        logs=json.load(open(AUDIT_LOG_PATH))
    if not logs: st.info("No logs yet"); return
    st.dataframe(pd.DataFrame(logs))
    st.download_button("Download CSV",pd.DataFrame(logs).to_csv(index=False),"audit_log.csv")

def page_settings_help():
    st.markdown("<div class='card'><h2>⚙️ Settings & Help</h2></div>",unsafe_allow_html=True)
    st.write("Password rules: min 8 chars, uppercase, lowercase, digit, special char")
    st.write("Theme: placeholder")
    st.write("Help: Use sidebar to navigate pages")

# -----------------------
# Login/Register Flow
# -----------------------
def page_login():
    st.markdown(
        """
        <style>
        .stApp {
          background: url('https://images.unsplash.com/photo-1465101046530-73398c7f28ca?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80')
          no-repeat center center fixed;
          background-size: cover;
        }

        .login-form-churn {
            width: 300px;
            margin: 5% auto;
            background: rgba(255, 255, 255, 0.12);
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.25);
            padding: 1rem 1rem;
            text-align: center;
            color: #f0f0f0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            border: 1px solid rgba(255,255,255,0.25);
            backdrop-filter: blur(12px);
            align-items: center;
            form-align: above;
            animation: fadeIn 1s ease-in-out;

        }

        .login-form-churn h2 {
            font-size: 26px;
            font-weight: 800;
            color: #00e0ff;
            margin-bottom: 12px;
            text-shadow: 0 0 8px rgba(255,255,255,0.7);
        }

        .stTextInput > div > div > input {
            width: 250px !important;
            background-color: rgba(255, 255, 255, 0.15);
            color: #fff;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.3);
            margin: 0 auto;
            padding: 6px 10px;
            font-size: 14px;
        }

        .stTextInput > label {
            color: #f1f1f1 !important;
            font-weight: 600 !important;
            font-size: 14px;
        }

        .stButton>button {
            width: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #45a049, #4CAF50);
            transform: scale(1.03);
        }

        .register-btn {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }

        .forgot-link {
            display: block;
            margin-top: 10px;
            color: #00e0ff;
            font-weight: 600;
            text-decoration: underline;
            cursor: pointer;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='login-form-churn'>
            <h2>🤖 AI-Powered Churn Prediction</h2>
            <div style='margin-bottom:18px;font-size:18px;font-weight:600;color:white;'>🔐 Admin Login</div>
        
        """,
        unsafe_allow_html=True,
    )
    
    # Render the form right after so it appears visually inside the box
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="Enter your email")
        pw = st.text_input("Password", type="password", placeholder="Enter your password")
        forgot = st.form_submit_button("Forgot Password?", use_container_width=False)
        submit = st.form_submit_button("Login")
        if forgot:
            st.session_state["page"] = "forgot_password"
        if submit:
            if _authenticate_admin(email, pw):
                st.session_state["auth"] = {"is_authenticated": True, "email": email, "role": "admin"}
                st.success("✅ Logged in successfully!")
                _log_action("Login",email)
                st.session_state['page']='dashboard'
            else:
                st.error("❌ Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Register"):
        st.session_state["page"] = "register"
   


def page_register():
    st.markdown(
        """
        <style>
         .stApp {
            background: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80') no-repeat center center fixed;
            background-size: cover;
        }
        .register-form-bg {
            width: 350px;
            margin: 6% auto;
            background: rgba(255,255,255,0.18);
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.18);
            padding: 1.5rem 1rem;
            text-align: center;
            color: #222;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            border: 1px solid rgba(255,255,255,0.25);
            backdrop-filter: blur(12px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='register-form-bg'>
            <img src="https://images.unsplash.com/photo-1519125323398-675f0ddb6308?auto=format&fit=crop&w=400&q=80" alt="Register" style="width:80px;height:80px;border-radius:50%;margin-bottom:12px;">
            <h2>📝 Register Admin</h2>
        """,
        unsafe_allow_html=True,
    )
    st.title("📝 Register Admin")
    with st.form("reg_form"):
        email=st.text_input("Email")
        pw=st.text_input("Password",type="password")
        confirm=st.text_input("Confirm Password",type="password")
        submit=st.form_submit_button("Create Account")
        if submit:
            if pw!=confirm: st.error("Passwords do not match")
            else:
                success,msg=_register_admin(email,pw)
                if success: st.success(msg); st.session_state["page"]="login"
                else: st.error(msg)
    if st.button("Back to Login"): st.session_state["page"]="login"

def page_forgot_password():
    st.title("🔑 Reset Password")
    with st.form("reset_form"):
        email = st.text_input("Enter your registered email")
        new_pw = st.text_input("New Password", type="password")
        confirm_pw = st.text_input("Confirm New Password", type="password")
        submit = st.form_submit_button("Reset Password")
        if submit:
            if new_pw != confirm_pw:
                st.error("Passwords do not match")
            else:
                success, msg = _reset_password(email, new_pw)
                if success:
                    st.success(msg)
                    st.session_state["page"] = "login"
                else:
                    st.error(msg)

    if st.button("Back to Login"):
        st.session_state["page"] = "login"


# -----------------------
# Styles
# -----------------------
def _inject_app_styles():
    st.markdown(
        """
        <style>
        .app-header { text-align:center; font-size:2rem; font-weight:800; color:#1f2937; margin:10px 0 16px; }
        .app-card { background:#ffffffcc; backdrop-filter:blur(6px); border-radius:16px; padding:20px; box-shadow:0 14px 30px rgba(0,0,0,0.12); max-width:1000px; margin:8px auto 24px; }
        .result-badge { margin-top:14px; text-align:center; padding:14px 18px; border-radius:12px; font-weight:800; color:#fff; }
        .result-badge.success { background: linear-gradient(90deg, #00b09b, #96c93d);}
        .result-badge.danger { background: linear-gradient(90deg, #ff416c, #ff4b2b);}
        button.stButton>button { background: linear-gradient(90deg,#7b2ff7,#f107a3)!important; color:white!important; border-radius:999px!important; font-weight:700;}
        </style>
        """,
        unsafe_allow_html=True
    )
def page_data_analysis():
    st.markdown("<div class='card'><h2>📊 Data Analysis</h2></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a CSV file for analysis", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Dataset Overview")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())

        # Missing Values
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing"}).astype(str))

        # Data Types
        st.subheader("Column Types")
        st.dataframe(df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Dtype"}).astype(str))

        # Target distribution (if exists)
        if "Churn" in df.columns or "churn" in df.columns:
            churn_col = "Churn" if "Churn" in df.columns else "churn"
            st.subheader("Churn Distribution")
            churn_counts = df[churn_col].value_counts()
            st.bar_chart(churn_counts)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Feature Distributions
        st.subheader("Feature Distributions")
        numeric_cols = numeric_df.columns.tolist()
        for col in numeric_cols[:5]:  # show first 5 to avoid too many plots
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)


def show_app_page():
    _inject_app_styles()
    st.sidebar.title("📊 Admin Panel")
    
    page_choice = st.sidebar.radio("Navigate", ["Dashboard", "Prediction History", "User Management", "Model Performance", "Data Upload", "Settings", "Help", "Audit Log"])

    if st.sidebar.button("Logout"):
        st.session_state["auth"] = {"is_authenticated": False, "email": ""}
        st.session_state["page"] = "login"
    # <- force the app to reload and show login page

        if page_choice == "Dashboard":
            page_dashboard()
        elif page_choice == "Prediction History":
            page_history()
        elif page_choice == "User Management":
            page_user_mgmt()
        elif page_choice == "Model Performance":
           _load_model_and_columns()
        elif page_choice == "Data Upload":
            page_batch_upload()
        elif page_choice == "Settings":
            page_settings_help()
        elif page_choice == "Help":
            page_settings_help()
        elif page_choice == "Audit Log":
            page_audit_log()
        elif page_choice == "logout":
            st.session_state["auth"]={"is_authenticated":False,"email":""}
            st.session_state["page"]="login"


# -----------------------
# Router
# -----------------------
inject_styles()
if "auth" not in st.session_state: st.session_state["auth"]={"is_authenticated":False,"email":""}
if "page" not in st.session_state: st.session_state["page"]="login"

if st.session_state["auth"]["is_authenticated"]:
    page_choice=sidebar_menu()
    if page_choice=="Dashboard": page_dashboard()
    elif page_choice=="Prediction History": page_history()
    elif page_choice=="User Management": page_user_mgmt()
    elif page_choice=="Model Performance": page_model_perf()
    elif page_choice=="Data Upload": page_batch_upload()
    elif page_choice == "Data Analysis": page_data_analysis()
    elif page_choice=="Audit Log": page_audit_log()
    elif page_choice=="Settings & Help": page_settings_help()
    elif page_choice=="Logout":
        st.session_state["auth"]={"is_authenticated":False,"email":""}
        st.session_state["page"]="login"
        
else:
    page=st.session_state["page"]
    if page=="login": page_login()
    elif page=="register": page_register()
    elif page == "forgot_password": page_forgot_password()
    else: page_login()
