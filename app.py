import streamlit as st
import pandas as pd
import joblib
import shap
import os
import datetime
import io
import numpy as np
import plotly.graph_objects as go
import random


st.set_page_config(page_title="پلتفرم هوشمند نگهداشت", layout="wide", page_icon="💎")

st.markdown("""
    <style>
    * { direction: rtl; text-align: right; font-family: 'Tahoma', sans-serif; }
    
    [data-testid="stAppViewContainer"] { 
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); 
        color: #f8fafc; 
    }
    [data-testid="stHeader"] { background-color: transparent; }
    
    html, body, [class*="st-"] { font-size: 32px !important; }
    
    .stSelectbox label, .stNumberInput label, .stRadio label, .stTextInput label, .stFileUploader label {
        font-size: 32px !important;
        color: #cbd5e1 !important;
        margin-bottom: 10px !important;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 50px; 
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
        margin-bottom: 40px;
    }
    
    .action-box { 
        background-color: rgba(139, 92, 246, 0.1); 
        color: #e2e8f0 !important; 
        padding: 36px; 
        border-radius: 20px; 
        border-right: 12px solid #8b5cf6; 
        margin-bottom: 24px; 
        font-size: 36px; 
        line-height: 1.8;
    }
    .reason-box { 
        background-color: rgba(245, 158, 11, 0.1); 
        color: #fcd34d !important; 
        padding: 30px; 
        border-radius: 20px; 
        border-right: 12px solid #f59e0b; 
        margin-bottom: 24px; 
        font-size: 36px; 
        font-weight: bold;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white !important;
        border: none;
        border-radius: 16px;
        padding: 24px 48px !important;
        font-size: 40px !important; 
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-4px); box-shadow: 0 8px 25px rgba(139, 92, 246, 0.4); }
    
    h1 { font-size: 76px !important; font-weight: 900 !important; margin-bottom: 10px !important; background: -webkit-linear-gradient(#fff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h3 { font-size: 48px !important; font-weight: bold !important; margin-bottom: 25px !important; color: #e2e8f0 !important;}
    h4 { font-size: 40px !important; }
    
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 40px; }
    .stTabs [data-baseweb="tab"] { font-size: 40px; color: #94a3b8; padding: 20px; }
    .stTabs [aria-selected="true"] { color: #8b5cf6 !important; border-bottom-color: #8b5cf6 !important; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load('churn_model.pkl')

@st.cache_data
def load_database():
    file_path = 'data/MCI_Challenge_FinalDataset.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

model = load_model()
db_data = load_database()


FEATURE_TRANSLATION = {
    'هزینه_ماهیانه_تومان': 'هزینه ماهیانه بالا', 'سابقه_سیم‌کارت_ماه': 'سابقه سیم‌کارت',
    'نوع_سیم‌کارت_اعتباری': 'اعتباری بودن سیم‌کارت', 'سرویس_تماس_VoLTE_خیر': 'عدم استفاده از مکالمه HD (VoLTE)',
    'فضای_ابری_اپراتور_خیر': 'عدم استفاده از فضای ابری', 'نسل_اینترنت_همراه_5G': 'تعرفه بالای اینترنت 5G',
    'نسل_اینترنت_همراه_2G': 'کیفیت پایین اینترنت 2G', 'استفاده_اپلیکیشن_اپراتور_خیر': 'عدم نصب اپلیکیشن مدیریت حساب',
    'هزینه_کل_تومان': 'مجموع پرداختی مشتری', 'بسته_اینترنت_شبانه_خیر': 'عدم استفاده از بسته شبانه',
    'سوپراپ_شبکه اجتماعی_خیر': 'عدم نصب شبکه اجتماعی', 'سوپراپ_خدمات_مالی_خیر': 'عدم استفاده از خدمات مالی',
    'بسته_رومینگ_بین‌الملل_خیر': 'عدم استفاده از رومینگ'
}

GENERAL_FALLBACKS = [
    "📞 ارجاع فوری به تیم نگهداشت (Call Center) برای بررسی دقیق و مصاحبه خروج.",
    "🎁 ارسال پیامک نظرسنجی با وعده اینترنت هدیه جهت کشف مشکل پنهان مشتری.",
    "📱 ارجاع پروفایل به کارشناس پشتیبانی ویژه (VIP) جهت تماس مستقیم.",
    "💡 ارسال کمپین پیامکی عمومی بازگشت با ۳۰٪ تخفیف روی تمام خدمات.",
    "🔄 پیشنهاد بررسی مجدد الگوی مصرف توسط سیستم هوشمند."
]

ACTION_CATALOG = {
    # -- هزینه ماهیانه --
    'هزینه_ماهیانه_تومان_1': '💰 ترکیب اینترنت و مکالمه با ۳۰٪ کمتر — یک بسته، همه چیز.',
    'هزینه_ماهیانه_تومان_2': '🎁 آخر هفته‌ها اینترنت مصرف کن، هزینه‌ای پرداخت نکن.',
    'هزینه_ماهیانه_تومان_3': '💵 ۲۰٪ از هر خرید بسته برمی‌گردد به کیف پولت — خرید بعدی ارزان‌تر.',
    'هزینه_ماهیانه_تومان_4': '🌙 ۷ شب اینترنت شبانه نامحدود — دانلود کن بدون نگرانی از حجم.',
    'هزینه_ماهیانه_تومان_101': '🛠️ بسته دلخواهت رو خودت بساز! حجم و مدت‌زمان را خودت انتخاب کن تا هزینه‌هات کاملاً مدیریت بشه.',
    'هزینه_ماهیانه_تومان_102': '🌙 اینترنت شبانه ۱۱ تا ۱ بامداد برگشت! مخصوص شما برای دانلودهای سنگین با کمترین هزینه.',
    'هزینه_ماهیانه_تومان_103': '⏱️ بسته شبانه یک‌روزه فعال شد — فقط برای امشب، با قیمتی که باورت نمی‌شه.',
    
    # -- سابقه سیم‌کارت --
    'سابقه_سیم‌کارت_ماه_1': '🎁 خوش آمدی — ۳ روز اینترنت رایگان برای شروع، بدون هیچ شرطی.',
    'سابقه_سیم‌کارت_ماه_2': '🔥 اولین بسته‌ات را بخر، دو برابرش تحویل بگیر — یک بار در طول اشتراک.',
    'سابقه_سیم‌کارت_ماه_3': '📞 خط اختصاصی پشتیبانی برای تو — بدون صف، بدون انتظار.',
    'سابقه_سیم‌کارت_ماه_4': '📺 مسابقه زنده می‌بینی؟ ترافیک تماشا از بسته‌ات کم نمی‌شود.',
    'سابقه_سیم‌کارت_ماه_101': '🎵 به پاس همراهی شما، امکان انتخاب آهنگ پیشواز دلخواه مخاطب برای شما فعال شد.',

    # -- نوع سیم‌کارت اعتباری --
    'نوع_سیم‌کارت_اعتباری_1': '🔄 سیم‌کارت دائمی = قبض ماهانه + ۳ ماه تخفیف + خیال راحت — همین الان تبدیل کن.',
    'نوع_سیم‌کارت_اعتباری_2': '📦 یک بار بخر، چند ماه استفاده کن — بسته بلندمدت با قیمت ثابت بدون نگرانی.',
    'نوع_سیم‌کارت_اعتباری_3': '🎁 شارژ خریدی؟ دو برابرش تحویل می‌گیری — یک بار در ماه.',
    'نوع_سیم‌کارت_اعتباری_4': '🌙 هر شب بعد از ساعت ۱۲، اینترنت رایگان است — بدون کسر از سهمیه روزانه‌ات.',
    'نوع_سیم‌کارت_اعتباری_101': '🔄 تبدیل سیم‌کارت اعتباری به دائمی با شرایط پرداخت اقساطی — بدون فشار مالی، از مزایای دائمی لذت ببر.',

    # -- عدم استفاده از VoLTE --
    'سرویس_تماس_VoLTE_خیر_1': '📞 VoLTE الان فعال می‌شود — تماس‌هایت واضح‌تر، بدون قطعی، حتی وسط اینترنت.',
    'سرویس_تماس_VoLTE_خیر_2': '🎧 ۷ روز تماس HD رایگان — تفاوتش را خودت بشنو، بعد تصمیم بگیر.',
    'سرویس_تماس_VoLTE_خیر_3': '📞 در مترو، مراکز خرید و جاهای شلوغ — VoLTE تماست را قطع نمی‌کند.',
    'سرویس_تماس_VoLTE_خیر_4': '🎮 بازی کن، همزمان تماس بگیر — VoLTE کیفیت را پایین نمی‌آورد.',
    'سرویس_تماس_VoLTE_خیر_101': '📞 آیا می‌دانستید با VoLTE کیفیت صدای شما در تماس‌ها چندین برابر می‌شود؟ همین الان آموزش فعال‌سازی را ببینید.',

    # -- فضای ابری --
    'فضای_ابری_اپراتور_خیر_1': '☁️ ۲۰ گیگ فضای ابری همین الان برای تو فعال می‌شود — عکس‌هایت را از دست نده.',
    'فضای_ابری_اپراتور_خیر_2': '🔒 اطلاعاتت روی سرورهای امن اپراتور ذخیره می‌شود — نه جاهای ناشناس.',
    'فضای_ابری_اپراتور_خیر_3': '📡 هنگام بکاپ گرفتن، اینترنت نیم‌بها است — داده‌هایت را با کمترین هزینه ذخیره کن.',
    'فضای_ابری_اپراتور_خیر_4': '🎬 کاربران Cloud اشتراک رایگان فیلم و سریال می‌گیرند — یک مزیت که نمی‌دانستی.',

    # -- اینترنت 5G --
    'نسل_اینترنت_همراه_5G_1': '🚀 سریع‌ترین اینترنت موبایل با قیمت معمولی — بسته 5G اقتصادی.',
    'نسل_اینترنت_همراه_5G_2': '🌙 هر ساعت یک سهمیه اینترنت 5G رایگان — برای وقتی که بسته‌ات تمام شده.',
    'نسل_اینترنت_همراه_5G_3': '📡 با 5G، تماس‌هایت هم بهتر می‌شود — صدا و اینترنت، هر دو با کیفیت.',
    'نسل_اینترنت_همراه_5G_101': '🎬 تماشای جدیدترین فیلم‌ها بدون نیاز به خرید اشتراک در فیلیمو، فیلم‌نت و نماوا — فقط با اینترنت همراه اول.',
    'نسل_اینترنت_همراه_5G_102': '🍿 دوست داری جدیدترین فیلم‌ها رو ببینی؟ همین الان بدون خرید اشتراک، رایگان تماشا کن!',

    # -- اینترنت 2G --
    'نسل_اینترنت_همراه_2G_1': '📳 سیم‌کارت 4G رایگان تحویل می‌گیری — همان شماره، سرعت چند برابر.',
    'نسل_اینترنت_همراه_2G_2': '🌐 هفته اول ارتقا، اینترنت رایگان داری — بدون هیچ شرطی.',
    'نسل_اینترنت_همراه_2G_3': '💰 گوشی 4G می‌خری؟ تخفیف ویژه برای مشترکین در حال ارتقا.',
    'نسل_اینترنت_همراه_2G_4': '📳 یوسیم رایگان درب خانه‌ات تحویل می‌گیری — نیازی به مراجعه نیست.',

    # -- عدم استفاده از اپلیکیشن --
    'استفاده_اپلیکیشن_اپراتور_خیر_1': '📲 اپلیکیشن را نصب کن، ۵ گیگ اینترنت رایگان بگیر — همین الان.',
    'استفاده_اپلیکیشن_اپراتور_خیر_2': '💰 از داخل اپ بسته بخر، ارزان‌تر تمام می‌شود — تخفیف اختصاصی کاربران اپ.',
    'استفاده_اپلیکیشن_اپراتور_خیر_3': '📲 ماموریت روزانه = جایزه روزانه — چند دقیقه در اپ، اینترنت رایگان.',
    'استفاده_اپلیکیشن_اپراتور_خیر_4': '💎 پکیج VIP کاربران اپ — بیشترین امکانات با قیمت ویژه.',
    'استفاده_اپلیکیشن_اپراتور_خیر_101': '🎁 امتیازات بلااستفاده داری! همین الان با ورود به اپلیکیشن، امتیازاتت رو به بسته‌های جذاب اینترنت تبدیل کن.',
    'استفاده_اپلیکیشن_اپراتور_خیر_102': '💡 هر روز یک ترفند و یک خدمت جدید — با ورود روزانه به اپلیکیشن، امکانات مخفی همراه اول را کشف کن.',
    'استفاده_اپلیکیشن_اپراتور_خیر_103': '🎟️ کوپن‌های همراه اول اینجاست! در این برنامه‌ها و فروشگاه‌ها، تخفیف‌های ویژه منتظر شماست.',
    
    # -- سرویس‌های جدید --
    'بسته_اینترنت_شبانه_خیر_1': '🌙 اختصاص یک بسته اینترنت شبانه رایگان ۳ روزه جهت تست کیفیت و سرعت.',
    'بسته_اینترنت_شبانه_خیر_2': '🦉 پیشنهاد بسته شبانه نامحدود با ۵۰٪ تخفیف اختصاصی (ویژه دانلود فایل‌های حجیم).',
    'بسته_اینترنت_شبانه_خیر_3': '🎬 ارسال پیامک: "فیلم‌های جدید رسید! امشب با اینترنت شبانه رایگان تماشا کن".',
    'سوپراپ_شبکه اجتماعی_خیر_1': '💬 تخصیص ترافیک رایگان و بدون محدودیت برای استفاده از شبکه‌های اجتماعی داخلی.',
    'سوپراپ_شبکه اجتماعی_خیر_2': '🎁 جایزه ویژه: با نصب و فعال‌سازی سوپراپ اجتماعی اپراتور، ۱۰ گیگ اینترنت دریافت کنید.',
    'سوپراپ_خدمات_مالی_خیر_1': '💳 تخصیص ۵۰ هزار تومان اعتبار اولیه رایگان در کیف پول سوپراپ مالی شما.',
    'سوپراپ_خدمات_مالی_خیر_2': '💸 با پرداخت اولین قبض از طریق سوپراپ مالی، قبض بعدی را مهمان ما باشید.',
    'بسته_رومینگ_بین‌الملل_خیر_1': '✈️ در حال برنامه‌ریزی سفر هستید؟ بسته‌های رومینگ اقتصادی ما را با تخفیف ویژه رزرو کنید.',
    'بسته_رومینگ_بین‌الملل_خیر_2': '🌍 سیم‌کارت شما برای استفاده بین‌المللی ارتقا یافت. (ارسال پیامک اطلاع‌رسانی تعرفه‌های جدید).'
}


def log_feedback(user_id, churn_prob, action_taken, feedback_status):
    log_file = 'MLOps_Action_Logs.csv'
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = pd.DataFrame([{'Timestamp': timestamp, 'User_ID': user_id, 'Predicted_Risk': churn_prob, 'Suggested_Action': action_taken, 'Status': feedback_status}])
    if not os.path.isfile(log_file): log_data.to_csv(log_file, index=False, encoding='utf-8-sig')
    else: log_data.to_csv(log_file, mode='a', header=False, index=False, encoding='utf-8-sig')

def draw_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        number = {'suffix': "%", 'font': {'size': 80, 'color': 'white'}}, 
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': "#ef4444" if probability >= 0.6 else ("#f59e0b" if probability >= 0.4 else "#10b981")},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.15)"},
                {'range': [40, 60], 'color': "rgba(245, 158, 11, 0.15)"},
                {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.15)"}],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'family': "Tahoma"}, height=450, margin=dict(l=10, r=10, t=20, b=10)) 
    return fig

def analyze_and_display(input_df, target_user_id="Manual"):
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']
    churn_prob = model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    
    col_chart, col_reasons = st.columns([1, 2], gap="large")
    
    with col_chart:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center; color:#cbd5e1;'>وضعیت خروج (Churn Risk)</h4>", unsafe_allow_html=True)
        st.plotly_chart(draw_gauge(churn_prob), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_reasons:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if churn_prob < 0.4:
            st.success("✅ مشتری در منطقه امن قرار دارد.")
            st.info("پیشنهاد سیستم: اجرای کمپین‌های بیش‌فروشی.")
        else:
            X_transformed = preprocessor.transform(input_df)
            feature_names = preprocessor.get_feature_names_out()
            clean_names = [f.replace('one_hot__', '').replace('standard__', '').replace('remainder__', '') for f in feature_names]
            
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            user_shap = shap_values[1][0] if isinstance(shap_values, list) else (shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0])
            
            shap_df = pd.DataFrame({'Feature': clean_names, 'SHAP_Value': user_shap}).sort_values(by='SHAP_Value', ascending=False)
            
            st.markdown("### 🔍 کالبدشکافی هوش مصنوعی (دلایل ریزش):")
            for i, row in enumerate(shap_df.head(2).iterrows(), 1):
                f_raw = row[1]['Feature']
                d_name = next((v for k, v in FEATURE_TRANSLATION.items() if k in f_raw), f_raw)
                st.markdown(f'<div class="reason-box">🔻 {d_name}</div>', unsafe_allow_html=True)
                
            st.markdown("### 🎯 بهترین اقدامات عملیاتی (NBA):")
            
            
            possible_actions = []
            
            for index, row in shap_df.iterrows():
                feature_name = row['Feature']
                matched_actions = [v for k, v in ACTION_CATALOG.items() if feature_name in k]
                
                if matched_actions:
                    possible_actions = matched_actions
                    break 
            
            if len(possible_actions) >= 2:
                selected_actions = random.sample(possible_actions, 2)
                for action in selected_actions:
                    st.markdown(f'<div class="action-box">{action}</div>', unsafe_allow_html=True)
            elif len(possible_actions) == 1:
                st.markdown(f'<div class="action-box">{possible_actions[0]}</div>', unsafe_allow_html=True)
            else:
                fallback = random.choice(GENERAL_FALLBACKS)
                st.markdown(f'<div class="action-box">⚠️ راهکار عمومی:<br>{fallback}</div>', unsafe_allow_html=True)
            
            f1, f2 = st.columns(2)
            if f1.button("✅ تایید و اجرای کمپین", key=f"yes_{target_user_id}"):
                log_feedback(target_user_id, round(churn_prob, 2), "تایید کمپین داینامیک", "Accepted")
                st.toast("بازخورد مثبت در سیستم MLOps ثبت شد.", icon="✔️")
            if f2.button("❌ راهکار مناسب نیست", key=f"no_{target_user_id}"):
                log_feedback(target_user_id, round(churn_prob, 2), "رد کمپین داینامیک", "Rejected")
                st.toast("بازخورد منفی جهت بازآموزی مدل ثبت شد.", icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)


st.title("💎 پلتفرم هوشمند نگهداشت مشتری (AI Retention)")
st.markdown("<p style='color:#94a3b8; font-size:36px; margin-bottom:40px;'>موتور تصمیم‌ساز مبتنی بر یادگیری ماشین و تحلیل رفتار مصرف‌کننده</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["👤 جستجو و تحلیل فردی (CRM)", "📂 پردازش گروهی مارکتینگ (Batch)"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 استعلام سریع مشتری")
    col_search, col_btn = st.columns([4, 1])
    search_id_str = col_search.text_input("شناسه مشترک را وارد کنید:", label_visibility="collapsed", placeholder="مثلاً 1024")
    
    if col_btn.button("تحلیل بلادرنگ ⚡", use_container_width=True):
        if db_data is not None:
            try:
                search_id = int(search_id_str)
                user_row = db_data[db_data['شناسه_مشترک'] == search_id]
                if user_row.empty: st.error("مشترکی یافت نشد!")
                else:
                    clean_df = user_row.drop(columns=['شناسه_مشترک', 'ریزش'], errors='ignore')
                    analyze_and_display(clean_df, target_user_id=str(search_id))
            except ValueError:
                st.error("لطفاً فقط عدد وارد کنید.")
        else: st.error("خطا در اتصال به پایگاه داده.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("➕ یا ویژگی‌های مشتری جدید را دستی وارد کنید"):
        col1, col2, col3 = st.columns(3)
        with col1:
            sim_type = st.selectbox("نوع سیم‌کارت", ["اعتباری", "دائمی"])
            net_gen = st.selectbox("نسل اینترنت", ["2G", "3G", "4G", "5G", "فاقد سرویس دیتا"])
            tenure = st.number_input("سابقه (ماه)", 0, 200, 12)
            monthly_cost = st.number_input("هزینه ماهیانه (تومان)", 0, value=9500000, step=500000)
            total_cost = st.number_input("هزینه کل (تومان)", 0, value=114000000, step=1000000)
        with col2:
            gender = st.selectbox("جنسیت", ["زن", "مرد"])
            age = st.number_input("سن", 10, 100, 30)
            birth_month = st.selectbox("ماه تولد", ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"])
            app_usage = st.selectbox("استفاده از اپلیکیشن", ["بله", "خیر"])
        with col3:
            volte = st.selectbox("سرویس VoLTE", ["بله", "خیر", "فاقد سرویس دیتا"])
            cloud = st.selectbox("فضای ابری", ["بله", "خیر", "فاقد سرویس دیتا"])
            night_net = st.selectbox("بسته شبانه", ["بله", "خیر", "فاقد سرویس دیتا"])
            roaming = st.selectbox("بسته رومینگ", ["بله", "خیر", "فاقد سرویس دیتا"])
            social_app = st.selectbox("سوپراپ اجتماعی", ["بله", "خیر", "فاقد سرویس دیتا"])
            finance_app = st.selectbox("سوپراپ مالی", ["بله", "خیر", "فاقد سرویس دیتا"])
        if st.button("تحلیل اطلاعات دستی 🚀"):
            input_df = pd.DataFrame({'جنسیت': [gender], 'سن': [age], 'ماه_تولد': [birth_month], 'سابقه_سیم‌کارت_ماه': [tenure], 'نسل_اینترنت_همراه': [net_gen], 'بسته_رومینگ_بین‌الملل': [roaming], 'فضای_ابری_اپراتور': [cloud], 'بسته_اینترنت_شبانه': [night_net], 'سرویس_تماس_VoLTE': [volte], 'سوپراپ_شبکه اجتماعی': [social_app], 'سوپراپ_خدمات_مالی': [finance_app], 'نوع_سیم‌کارت': [sim_type], 'استفاده_اپلیکیشن_اپراتور': [app_usage], 'هزینه_ماهیانه_تومان': [monthly_cost], 'هزینه_کل_تومان': [total_cost]})
            analyze_and_display(input_df, target_user_id="Manual_Entry")

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📥 اجرای کمپین روی فایل داده (Batch Processing)")
    uploaded_file = st.file_uploader("فایل CSV مشتریان را بکشید و اینجا رها کنید", type="csv")
    
    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)
        st.info(f"فایل با موفقیت خوانده شد. تعداد کل مشترکین: {len(df_batch):,} نفر")
        
        if st.button("⚙️ شروع پردازش و شناسایی ریسک"):
            with st.spinner('مدل در حال تحلیل رکوردها است...'):
                X_batch = df_batch.drop(columns=['شناسه_مشترک', 'ریزش'], errors='ignore')
                batch_probs = model.predict_proba(X_batch)[:, 1]
                
                result_df = df_batch.copy()
                result_df['احتمال_ریزش_درصد'] = np.round(batch_probs * 100, 1)
                result_df['وضعیت_ریسک'] = result_df['احتمال_ریزش_درصد'].apply(lambda x: '🔴 پرخطر' if x >= 60 else ('🟡 متوسط' if x >= 40 else '🟢 کم‌خطر'))
                
                high_risk_count = len(result_df[result_df['احتمال_ریزش_درصد'] >= 60])
                
                st.markdown("---")
                st.warning(f"🚨 هشدار: **{high_risk_count:,} نفر** از مشتریان این فایل در وضعیت پرخطر قرار دارند!")
                
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label="📥 دانلود فایل خروجی",
                    data=csv_buffer.getvalue(),
                    file_name=f"Marketing_Campaign_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    st.markdown('</div>', unsafe_allow_html=True)