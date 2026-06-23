import streamlit as st
import pandas as pd 
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

#Webpage config
st.set_page_config(page_title = 'SME Credit Rating', layout = 'wide')
st.title('Đánh giá rủi ro tín dụng doanh nghiệp SME bằng trí tuệ nhân tạo')

#Create a list for query history
if 'credit_history' not in st.session_state:
    st.session_state.credit_history = []

#Load ML model and data
@st.cache_resource
def load_assets():
    with open('sme_model_assets.pkl', 'rb') as f:
        assets = pickle.load(f)
        return assets
    
assets = load_assets()
model = assets['model']
encoded_features_column = assets['encoded_features_column']
industries = assets['industries']
provinces = assets['provinces']

#Sidebar and user input form
st.sidebar.header('Nhập dữ liệu doanh nghiệp (đơn vị triệu VND)')
input = {}

input['Industry'] = st.sidebar.selectbox('Ngành nghề kinh doanh', industries)
input['Province'] = st.sidebar.selectbox('Tỉnh/Thành phố', provinces)
input['Firm_Age'] = st.sidebar.number_input('Số năm hoạt động', min_value = 0)
input['Revenue'] = st.sidebar.number_input('Doanh thu thuần năm gần nhất', min_value = 0)
input['Net_Profit'] = st.sidebar.number_input('Lợi nhuận sau thuế')
input['Total_Assets'] = st.sidebar.number_input('Tổng tài sản', min_value = 1)
input['Total_Liabilities'] = st.sidebar.number_input('Tổng nợ phải trả', min_value = 0)
input['Current_Assets'] = st.sidebar.number_input('Tài sản ngắn hạn', min_value = 0)
input['Current_Liabilities'] = st.sidebar.number_input('Nợ ngắn hạn', min_value = 0)
input['Operating_Cashflow'] = st.sidebar.number_input('Dòng tiền từ hoạt động kinh doanh')
input['Interest_Expense'] = st.sidebar.number_input('Chi phí lãi vay', min_value = 0)
input['EBIT'] = st.sidebar.number_input('Lợi nhuận trước lãi vay và thuế (EBIT)')
input['Collateral_Value'] = st.sidebar.number_input('Giá trị tài sản bảo đảm', min_value = 0)
input['Loan_Amount'] = st.sidebar.number_input('Khoản vay đề nghị', min_value = 0)
input['Credit_History_Years'] = st.sidebar.number_input('Số năm có lịch sử tín dụng', min_value = 0)
input['Late_Payment_Count'] = st.sidebar.number_input('Số lần trả chậm trong 12 tháng gần nhất', min_value = 0)
input['Tax_Compliance_Score'] = st.sidebar.slider('Điểm tuân thủ thuế', 0, 100)
input['Digital_Transaction_Share'] = st.sidebar.slider('Tỷ lệ giao dịch số', 0.0, 1.0)

#Start processing input data
if st.sidebar.button("Bắt đầu thẩm định"):
    #Calculate remaining values using input data
    equity = input['Total_Assets'] - input['Total_Liabilities']
    input['Equity'] = equity
    input['Current_Ratio'] = input['Current_Assets'] / input['Current_Liabilities']
    input['Debt_Ratio'] = input['Total_Liabilities'] / input['Total_Assets']
    input['ROA'] = (input['Net_Profit'] / input['Total_Assets']) * 100
    input['ROE'] = (input['Net_Profit'] / equity) * 100 if equity != 0 else 0
    input['Interest_Coverage'] = input['EBIT'] / input['Interest_Expense']
    input['Loan_to_Value'] = input['Loan_Amount'] / input['Collateral_Value']

    #Process input
    input_df_raw = pd.DataFrame([input])
    input_df_encoded = pd.get_dummies(input_df_raw, columns = ['Industry', 'Province'])
    input_df = input_df_encoded.reindex(columns = encoded_features_column, fill_value = 0)

    #Predict default probability
    default_proba = model.predict_proba(input_df)[0][1]

    #Calculate credit score
    credit_score = int(300 + (1 - default_proba) * 550)

    #Process credit score
    if credit_score >= 720:
        risk_class = 'Thấp'
        color = 'green'
    elif credit_score >= 520:
        risk_class = 'Trung bình'
        color = 'orange'
    else:
        risk_class = 'Cao'
        color = 'red'

    #Create a dataframe for company information
    display_df = input_df_raw.T
    display_df.columns = ['Giá trị']
    display_df.index.name = 'Danh mục'

    #Process values for displaying
    def format_value(val):
        if isinstance(val, float):
            return f'{val:.2f}'
        elif isinstance(val, (int, np.integer)):
            return f'{val:,}'
        else:
            return str(val)

    display_df['Giá trị'] = display_df['Giá trị'].apply(format_value)

    #Table for company data
    st.subheader('Thông tin doanh nghiệp')
    st.dataframe(display_df, width = 'stretch')

    st.divider()

    #Displaying results
    st.subheader('Kết quả đánh giá rủi ro')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label = 'Điểm tín dụng (Credit Score)', value = f'{credit_score} / 850')
    with col2:
        st.metric(label = 'Xác suất Vỡ nợ', value = f'{default_proba * 100}%')
    with col3:
        st.markdown(f"Phân loại rủi ro: <br><b style = 'color: {color}; font-size: 22px;'>{risk_class}</b>", unsafe_allow_html = True)

    st.divider()

    #SHAP table
    st.subheader('Phân tích các yếu tố tác động lớn nhất đến kết quả dự đoán sử dụng SHAP')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        current_shap_values = shap_values[1][0]
    else:
        current_shap_values = shap_values[..., 1][0] if len(shap_values.shape) == 3 else shap_values[0]
    
    fig, ax = plt.subplots(figsize = (10, 6))
    exp = shap.Explanation(values = current_shap_values, data = input_df.values[0], feature_names = encoded_features_column)
    shap.plots.bar(exp, max_display = 10, show = False)
    plt.tight_layout()
    st.pyplot(fig)

    #Save record for query history
    time = pd.Timestamp.now(tz = 'UTC').tz_convert('Asia/Ho_Chi_Minh')
    new_record = {
        "Thời gian": time.strftime('%H:%M:%S'),
        'Ngành nghề kinh doanh': input['Industry'],
        'Tỉnh/Thành phố': input['Province'],
        'Điểm tín dụng': credit_score,
        'Xác suất vỡ nợ': f'{default_proba * 100}%',
        'Phân loại rủi ro': risk_class
    }
    st.session_state.credit_history.insert(0, new_record)
else:
    st.info('Hãy sử dụng menu bên trái để điều chỉnh các thông số của doanh nghiệp sau đó bấm nút "Bắt đầu thẩm định".')

st.divider()

#Displaying query history
st.subheader('Lịch sử các lần đánh giá')
if st.session_state.credit_history:
    history_df = pd.DataFrame(st.session_state.credit_history)
    st.dataframe(history_df, width = 'stretch')

    if st.button('Xoá toàn bộ lịch sử'):
        st.session_state.credit_history = []
        st.rerun()
else:
    st.caption('Chưa thực hiện lượt đánh giá nào trong phiên làm việc này.')
