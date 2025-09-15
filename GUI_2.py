import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import re
from pyvi.ViTokenizer import tokenize
from pyvi import ViTokenizer
from wordcloud import WordCloud

STOP_WORD_FILE = 'vietnamese-stopwords.txt'
teen_code_file = 'teencode.txt'
english_vnmese_file = 'english-vnmese.txt'
wrong_word_file = 'wrong-word.txt'
emoji_file = 'emojicon.txt'

# đọc stopwords (1 từ mỗi dòng)
with open(STOP_WORD_FILE, 'a', encoding='utf-8') as f:
        f.write(f'\nkhách_sạn')

with open(STOP_WORD_FILE, 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f if line.strip()])

def load_dict_from_txt(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(",")  # tách theo dấu phẩy
            if len(parts) == 2:
                mapping[parts[0].strip()] = parts[1].strip()
    return mapping

# load các từ điển thay thế
teen_dict = load_dict_from_txt(teen_code_file)
en_vi_dict = load_dict_from_txt(english_vnmese_file)
wrong_dict = load_dict_from_txt(wrong_word_file)
emoji_dict = load_dict_from_txt(emoji_file)

def preprocess_text(text, remove_stopwords=True):
    if pd.isna(text):
        return ["unk"]   # fallback cho dữ liệu trống

    text = text.lower().strip()

    # thay teen code
    for k, v in teen_dict.items():
        text = re.sub(r'\b{}\b'.format(re.escape(k)), v, text)

    # dịch từ tiếng Anh sang Việt
    for k, v in en_vi_dict.items():
        text = re.sub(r'\b{}\b'.format(re.escape(k)), v, text)

    # sửa chính tả
    for k, v in wrong_dict.items():
        text = re.sub(r'\b{}\b'.format(re.escape(k)), v, text)

    # chuyển emoji
    for k, v in emoji_dict.items():
        text = text.replace(k, v)

    # bỏ ký tự đặc biệt & số
    text = re.sub(r'[^a-zA-ZÀ-ỹ\s]', ' ', text)

    # tách từ bằng PyVi
    text = ViTokenizer.tokenize(text)

    # tách từ
    tokens = text.split()

    # loại stopwords
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stopwords]

    # fallback nếu tokens rỗng
    if len(tokens) == 0:
        return ["unk"]

    return tokens

#df['sentiment'] = df.Star.apply(lambda x: 1 if x >=4 else 0)
def plt_word_cloud_tabs(df, hotel_id):
    hotel_df = df[df["Hotel_ID"] == hotel_id].copy()
    hotel_df['Sentiment'] = hotel_df['Score'].apply(lambda x: 1 if x >= 7 else 0)

    positive_reviews = ' '.join(hotel_df[hotel_df['Sentiment'] == 1]['Body'])
    negative_reviews = ' '.join(hotel_df[hotel_df['Sentiment'] == 0]['Body'])

    positive_words = preprocess_text(positive_reviews) if positive_reviews else ["No positive reviews"]
    negative_words = preprocess_text(negative_reviews) if negative_reviews else ["No negative reviews"]

    # Positive
    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(positive_words))
    fig_pos, ax_pos = plt.subplots(figsize=(8, 6))
    ax_pos.imshow(positive_wordcloud, interpolation='bilinear')
    ax_pos.set_title('Positive Reviews')
    ax_pos.axis('off')

    # Negative
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(negative_words))
    fig_neg, ax_neg = plt.subplots(figsize=(8, 6))
    ax_neg.imshow(negative_wordcloud, interpolation='bilinear')
    ax_neg.set_title('Negative Reviews')
    ax_neg.axis('off')

    return fig_pos, fig_neg


def recommend_hotels(df, hotel_id):
    detail_cols = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']
    
    # Điểm trung bình của khách sạn
    hotel_scores = df[df["Hotel_ID"] == hotel_id][detail_cols].mean()
    
    # Trung bình toàn bộ khách sạn
    global_avg = df[detail_cols].mean()

    # So sánh chi tiết
    recommendations = []
    for col in detail_cols:
        score = hotel_scores[col]
        avg = global_avg[col]

        if score < avg:
            if col == "Facilities":
                recommendations.append("🏗 **Cơ sở vật chất** dưới mức trung bình → Cần bảo trì phòng ốc, cải thiện tiện ích chung (wifi, thang máy...).")
            elif col == "Cleanliness":
                recommendations.append("🧹 **Độ sạch sẽ** chưa đạt chuẩn → Tăng cường kiểm tra vệ sinh phòng & khu vực công cộng.")
            elif col == "Service":
                recommendations.append("🙋 **Dịch vụ** kém hơn trung bình → Đào tạo nhân viên về thái độ & kỹ năng hỗ trợ khách.")
            elif col == "Value_for_money":
                recommendations.append("💵 **Giá trị đồng tiền** chưa tốt → Xem xét lại chính sách giá & bổ sung ưu đãi.")
            elif col == "Location":
                recommendations.append("📍 **Vị trí** không nổi bật → Cung cấp thêm dịch vụ đưa đón hoặc hướng dẫn chi tiết khu vực xung quanh.")
        else:
            recommendations.append(f"✅ **{col}** tốt hơn trung bình ({score:.1f} vs {avg:.1f}).")

    # Tổng kết
    if sum(hotel_scores < global_avg) >= 3:
        summary = "⚠️ Nhiều tiêu chí dưới mức trung bình. **Cần cải thiện để tốt hơn!** ❌"
    elif sum(hotel_scores > global_avg) >= 3:
        summary = "👉 Khách sạn có nhiều tiêu chí vượt mức trung bình. **Rất Tôt, Cứ thế mà phát huy** ✅"
    else:
        summary = "🤔 Điểm số cân bằng với mức chung. **Tạm ổn, nhưng cần cải thiện để tốt hơn.**"

    # Xuất ra Streamlit
    st.subheader("📝 Lời khuyên cải thiện cho khách sạn")
    for rec in recommendations:
        st.markdown("- " + rec)
    st.info(summary)


# Using menu
st.title("Recommender System")
st.image("agoda.png", width=500)
st.image("agoda2.png", width=500)
menu = ["Home", "Project Understanding", "Gợi Ý Khách Sạn", "Hotel Insights"]
choice = st.sidebar.selectbox('Menu', menu)


if choice == 'Home':    
    st.subheader("[Hotel Booking](https://www.agoda.com/?gclsrc=aw.ds&gad_source=1&gad_campaignid=22632286545&gbraid=0AAAAAo6DVViHXFHCbA1gm3eSLklRmqqqt&gclid=Cj0KCQjw5onGBhDeARIsAFK6QJbKYS_CyocdKiR5S-Tc_093efMiiYTCljq30klcCp_pDyuLv22GLRgaAl0rEALw_wcB&cid=1844104&ds=vhWYN3PjCSmcWc8t)")  
elif choice == 'Project Understanding':    
    st.write("""# Triển khai Hệ thống Gợi ý cho Agoda nhằm:

### 1. Đề xuất các khách sạn/resort phù hợp nhất dựa trên lịch sử tìm kiếm, hành vi đặt phòng và đánh giá của người dùng.

### 2. Nâng cao trải nghiệm khách hàng, cải thiện tỷ lệ chuyển đổi đặt phòng và tăng mức độ hài lòng.

### 3. Cung cấp insights cho chủ khách sạn bao gồm:

 - Phân khúc khách hàng (theo độ tuổi, quốc gia, mục đích chuyến đi).

 - Xu hướng hành vi đặt phòng (thời điểm đặt, thời gian lưu trú, loại phòng ưa thích).

 - Tóm tắt đánh giá & phản hồi chính từ khách hàng.

 - Dự báo nhu cầu & khuyến nghị chiến lược giá/khuyến mãi.
             """)
    # hiển thị các hình ảnh liên quan đến đồ án
    st.image("photo.png", width=400, caption="### CF & CBF")
    st.image("insights.jpg", width=400, caption="### Hybrid")

############################################################################################################3  
#        
elif choice == 'Gợi Ý Khách Sạn':
            # function cần thiết
    def get_recommendations(df, hotel_id, cosine_sim, nums=5):
        # Get the index of the hotel that matches the hotel_id
        matching_indices = df.index[df['Hotel_ID'] == hotel_id].tolist()
        if not matching_indices:
            print(f"No hotel found with ID: {hotel_id}")
            return pd.DataFrame()  # Return an empty DataFrame if no match
        idx = matching_indices[0]

        # Get the pairwise similarity scores of all hotels with that hotel
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the hotels based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the nums most similar hotels (Ignoring the hotel itself)
        sim_scores = sim_scores[1:nums+1]

        # Get the hotel indices
        hotel_indices = [i[0] for i in sim_scores]

        # Return the top n most similar hotels as a DataFrame
        return df.iloc[hotel_indices]

    # Hiển thị đề xuất ra bảng
    def display_recommended_hotels(recommended_hotels, cols=5):
        for i in range(0, len(recommended_hotels), cols):
            cols = st.columns(cols)
            for j, col in enumerate(cols):
                if i + j < len(recommended_hotels):
                    hotel = recommended_hotels.iloc[i + j]
                    with col:   
                        st.write(hotel['Hotel_Name'])                    
                        expander = st.expander(f"Description")
                        hotel_description = hotel['Hotel_Description']
                        truncated_description = ' '.join(hotel_description.split()[:100]) + '...'
                        expander.write(truncated_description)
                        expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")   

    def truncate_text(text, max_words=50):
        """
        Cắt mô tả chỉ lấy tối đa max_words từ, thêm "..." nếu quá dài.
        """
        words = str(text).split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + " ..."
        return text

    def display_recommendations_markdown(recommendations):
        """
        Hiển thị danh sách khách sạn bằng markdown, giới hạn mô tả tối đa 50 từ.
        Thêm số thứ tự cho từng khách sạn.
        """
        for idx, row in enumerate(recommendations.itertuples(), start=1):
            desc = truncate_text(row.Hotel_Description, max_words=50)
            st.markdown(f"""
            ---
            ### {idx}. 🏨 {row.Hotel_Name}
            **ID:** {row.Hotel_ID}  
            **Mô tả:** {desc}  
            {"⭐ **Similarity Score:** {:.4f}".format(row.similar_score) if 'similar_score' in recommendations.columns else ""}
            """, unsafe_allow_html=True)

    # Đọc dữ liệu khách sạn
    df_hotels = pd.read_csv('hotel_info.csv')
    # Lấy 20 khách sạn
    unique_hotels = df_hotels.drop_duplicates(subset=["Hotel_ID"])
    random_hotels = unique_hotels.head(20)

    st.session_state.random_hotels = random_hotels

    # Open and read file to cosine_sim_new
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim_new = pickle.load(f)

    ###### Giao diện Streamlit ######
    #st.image('hotel.jpg', use_column_width=True)
    st.image("hotel.jpg", use_container_width=True)

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_hotel_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_hotel_id = None

    # Theo cách cho người dùng chọn khách sạn từ dropdown
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]
    # st.session_state.random_hotels
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn khách sạn",
        options=hotel_options,
        format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_hotel_id = selected_hotel[1]

    if st.session_state.selected_hotel_id:
        st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
        # Hiển thị thông tin khách sạn được chọn
        selected_hotel = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]
        

        if not selected_hotel.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', selected_hotel['Hotel_Name'].values[0])

            hotel_description = selected_hotel['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Top 5 khách sạn khác bạn cũng có thể quan tâm:')
            recommendations = get_recommendations(
                df_hotels, 
                st.session_state.selected_hotel_id, 
                cosine_sim=cosine_sim_new, 
                nums=5
            )

            display_recommendations_markdown(recommendations)
 
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")


#############################################################################################################
elif choice=='Hotel Insights':
    st.write("##### Hotel Insights")
    st.write("##### List Hotels")
    # Tạo dataframe hiển thị danh sách khách sạn
    df = pd.read_csv('hotel_info_temp.csv', index_col=False)

    st.title("Danh sách khách sạn")

    # Hiển thị danh sách khách sạn trong selectbox
    hotel_choice = st.selectbox("Chọn một khách sạn:", df['Hotel_Name'].unique())

    # Lọc thông tin khách sạn đã chọn
    selected_hotel = df[df['Hotel_Name'] == hotel_choice].iloc[0]

    # In ra thông tin chi tiết
    st.subheader("🏨 Thông tin chi tiết khách sạn")
    st.write(f"🆔 **ID:** {selected_hotel['Hotel_ID']}")
    st.write(f"📛 **Tên:** {selected_hotel['Hotel_Name']}")
    st.write(f"📍 **Địa chỉ:** {selected_hotel['Hotel_Address']}")
    st.write(f"⭐ **Điểm số:** {selected_hotel['Total_Score']}")
    st.write(f"🏅 **Hạng sao:** {selected_hotel['Hotel_Rank']}")
        # hotel_id là ID khách sạn bạn chọn ở selectbox/slider v.v.
    hotel_df = df[df["Hotel_ID"] == selected_hotel['Hotel_ID']]
    st.subheader("Reviewer theo quốc gia")
    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(18, 6))  # dùng fig, ax thay vì plt trực tiếp
    nationality_counts = hotel_df["Nationality"].value_counts()

    sns.barplot(x=nationality_counts.index, y=nationality_counts.values, ax=ax)
    ax.set_title("Reviewer Nationalities")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Hiển thị trên Streamlit
    st.pyplot(fig)
    

    def plt_comment_score_month(df, hotel_id):
        # Lọc theo Hotel_ID và năm >= 2023
        hotel_df = df[(df["Hotel_ID"] == hotel_id) & (df["Year"] >= 2023)]

        # Groupby để tính số lượng review và điểm trung bình
        monthly_stats = (
            hotel_df.groupby(["Year", "Month"])
            .agg(Review_Count=("Reviewer_ID", "count"),
                Avg_Score=("Score", "mean"))
            .reset_index()
        )

        # Tạo cột YearMonth dạng datetime chuẩn
        monthly_stats["YearMonth"] = pd.to_datetime(
            monthly_stats["Year"].astype(int).astype(str) + "-" +
            monthly_stats["Month"].astype(int).astype(str).str.zfill(2) + "-01",
            format="%Y-%m-%d"
        )

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Vẽ bar
        ax1.bar(
            monthly_stats["YearMonth"],
            monthly_stats["Review_Count"],
            color="skyblue", width=20
        )
        ax1.set_ylabel("Number of Reviews", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Vẽ lineplot
        ax2 = ax1.twinx()
        sns.lineplot(
            data=monthly_stats,
            x="YearMonth", y="Avg_Score",
            marker="o", color="red", linewidth=2, ax=ax2
        )
        ax2.set_ylabel("Average Score", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        ax1.set_xlabel("Time (Month)")
        plt.title(f"Hotel {hotel_id} - Reviews & Average Score (2023–2024)")
        plt.xticks(rotation=45)

        plt.tight_layout()

        return fig   # trả về fig thay vì plt.show()
    st.subheader("Số lượng review và điểm trung bình theo tháng")
    # Đọc dữ liệu review
    fig = plt_comment_score_month(df, hotel_id=selected_hotel['Hotel_ID'])
    st.pyplot(fig)

    def plt_detail_score(df, hotel_id):
        hotel_df = df[df["Hotel_ID"] == hotel_id]

        # các cột chi tiết
        detail_cols = ['Location', 'Cleanliness', 'Service', 'Facilities', 'Value_for_money']

        # trung bình theo khách sạn cụ thể
        detail_scores = hotel_df[detail_cols].mean()

        # trung bình toàn bộ khách sạn
        global_avg = df[detail_cols].mean()


        # vẽ bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        detail_scores.plot(kind='bar', color='skyblue', ax=ax, label=f'Hotel {hotel_id}')

        # vẽ line của trung bình toàn bộ
        ax.plot(global_avg.values, color='red', marker='o', linestyle='--', label='Global Average')

        ax.set_title(f'Average Score by Detail (Hotel {hotel_id}) vs Global')
        ax.set_xlabel('Detail')
        ax.set_ylabel('Average Score')
        ax.set_xticks(range(len(detail_cols)))
        ax.set_xticklabels(detail_cols, rotation=45)
        ax.set_ylim(0, 10)  # thang điểm 0-10
        ax.legend()
        return fig
    st.subheader("Điểm chi tiết so với trung bình toàn bộ khách sạn")
    fig = plt_detail_score(df, hotel_id=selected_hotel['Hotel_ID'])
    st.pyplot(fig)

    # đánh giá tích cực và tiêu cực
    st.subheader("Đánh giá tích cực và tiêu cực")
    
    # Tạo và hiển thị word cloud
        
    positive_fig, negative_fig = plt_word_cloud_tabs(df, hotel_id=selected_hotel['Hotel_ID'])

    tab1, tab2 = st.tabs(["Điểm Mạnh", "Cần Cải Thiện"])

    with tab1:
        st.pyplot(positive_fig)

    with tab2:
         st.pyplot(negative_fig)
         
    recommend_hotels(df, hotel_id=selected_hotel['Hotel_ID'])


        
        

    



