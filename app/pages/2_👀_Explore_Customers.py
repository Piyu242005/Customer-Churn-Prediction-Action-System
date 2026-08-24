import streamlit as st
st.set_page_config(page_title="Explore Customers",page_icon="👀",layout="wide")
st.title("👀 Explore Customers")
st.write("Browse and understand your customer records in a simple way.")
df=st.session_state.get("customer_data")
if df is None:
    st.info("No customer data is loaded yet. Open **📂 Upload Data** first or use Guest Demo on Home.")
    st.stop()
q=st.text_input("🔍 Search customers",placeholder="Type a customer name, ID, or value")
view=df.copy()
if q:view=view[view.astype(str).apply(lambda r:r.str.contains(q,case=False,na=False).any(),axis=1)]
a,b=st.columns(2);a.metric("Customers",f"{len(df):,}");b.metric("Columns",f"{len(df.columns):,}")
st.dataframe(view,use_container_width=True)
st.download_button("⬇️ Download this data",view.to_csv(index=False),"customers.csv","text/csv")
