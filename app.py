import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

train_df=pd.read_csv('./dataset/Train.csv')
cleaned_train=pd.read_csv('./dataset/cleanedTrain.csv')
item_ind_df=pd.read_csv('./dataset/clean_train.csv')
cleantrain=pd.read_csv('./dataset/cleantrain.csv')
model_result=pd.read_csv('./dataset/model_result.csv')



st.sidebar.header('DETAILS AND VISUALIZATION ABOUT STORE SALES :')
st.sidebar.title('CONTENTS')
res=st.sidebar.selectbox('CHOOSE TOPICS',['IMPUTATION','NUMERICAL-PLOTTING','CATEGORICAL-PLOTTING','MODEL-COMPARISON'])


if res=='IMPUTATION':
    res_imp=st.sidebar.selectbox('CHOOSE ONE',['Item_Weight','Outlet_Size'])
    

elif res=='NUMERICAL-PLOTTING':

    res_dis=st.sidebar.selectbox('CHOOSE ONE',['DIST-PLOT','SCATTER-PLOT','BOX-PLOT'])
    
    if res_dis=="DIST-PLOT":
        columns=st.sidebar.selectbox("CHOOSE COLUMN",['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Item_Outlet_Sales'])
    
    elif res_dis=="SCATTER-PLOT":
        columns1=st.sidebar.selectbox("CHOOSE FIRST COLUMN",['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Item_Outlet_Sales'])
        columns2=st.sidebar.selectbox("CHOOSE SECOND COLUMN",['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Item_Outlet_Sales'])
    
    elif res_dis=="BOX-PLOT":
        column_box=st.sidebar.selectbox("CHOOSE FIRST COLUMN",['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Item_Outlet_Sales'])

elif res=="CATEGORICAL-PLOTTING":
    res_dis=st.sidebar.selectbox('CHOOSE ONE',['COUNT-PLOT','PIE-CHART'])

    if res_dis in ['COUNT-PLOT','PIE-CHART','BAR-PLOT']:
        columns=st.sidebar.selectbox("CHOOSE COLUMN",['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type'])


else:
    res_dis=st.sidebar.selectbox('CHOOSE ONE',['R2-SCORE','MEAN-SQUARE-ERROR','ROOT-MEAN-SQUARE-ERROR','MEAN-ABSOLUTE-ERROR','SCORE-COMPARISON'])


show_details=st.sidebar.button('SHOW DETAILS')
st.sidebar.write('feel free to contact(email) : pravatpatra1997@gmail.com')

#for imputation:
if show_details:
    if res=='IMPUTATION':
        if res_imp=='Item_Weight':
            col1,col2=st.columns(2)
            with col1:
                st.subheader("BEFORE IMPUTATION")
                st.write("For example I have taken 'FDN15' as Item_Identifier")
                temp_df=train_df[train_df['Item_Identifier']=='FDN15'][["Item_Identifier","Item_Weight"]]
                st.dataframe(temp_df)
                fig,ax1=plt.subplots()
                sns.distplot(train_df['Item_Weight'],ax=ax1)
                ax1.set_title("Distribution of Item_Weight Before Imputation")
                st.pyplot(fig)
            

            with col2:
                st.subheader("AFTER IMPUTATION")
                st.write("For example I have taken 'FDN15' as Item_Identifier")
                temp_df=item_ind_df[item_ind_df['Item_Identifier']=='FDN15'][["Item_Identifier","Item_Weight2"]]
                st.dataframe(temp_df)
                fig,ax2=plt.subplots()
                sns.distplot(cleaned_train['Item_Weight'],ax=ax2,color='green')
                ax2.set_title("Distribution of Item_Weight After Imputation")
                st.pyplot(fig)

            st.markdown("**CODE USING FOR IMPUTATION :**")
            st.code("""
                    item_dic={}
                    for i in list(train_df['Item_Identifier'].unique()):
                        item_dic[i]=round(train_df[train_df['Item_Identifier']==i]['Item_Weight'].mean(),2)
                    """)
            st.code("""
                    train_df['Item_Weight']=train_df['Item_Weight'].fillna('missing')
                    weight=[]
                    for i in range(len(train_df)):
                        if train_df['Item_Weight'][i]=='missing':
                            val=round(item_dic[train_df['Item_Identifier'][i]],2)
                            weight.append(val)
                        else:
                            weight.append(train_df['Item_Weight'][i])
                    train_df['Item_Identifier']=weight
                    """)
            
            st.markdown("#### WHAT IF WE IMPUTE WITH MEAN AND MEDIAN :")
            col3,col4=st.columns(2)
            with col3:
                st.markdown("**IMPUTATION WITH MEAN**")
                temp_mean=train_df['Item_Weight'].fillna(train_df['Item_Weight'].mean())
                fig,ax1=plt.subplots()
                sns.distplot(temp_mean,ax=ax1,color='red')
                ax1.set_title("IMPUTATION WITH MEAN")
                st.pyplot(fig)

            with col4:
                st.markdown("**IMPUTATION WITH MEAN**")
                temp_median=train_df['Item_Weight'].fillna(train_df['Item_Weight'].median())
                fig,ax1=plt.subplots()
                sns.distplot(temp_median,ax=ax1,color='red')
                ax1.set_title("IMPUTATION WITH MEDIAN")
                st.pyplot(fig)

            st.write("We can't do this simply because it changes the distribution of 'Item_Weight' column.")
            st.markdown("**That's all about the imputation of `Item_Weight` column.**")




    #for 'Outlet_Size' column:
    #if show_details:
        if res_imp=='Outlet_Size':

            st.write("Here we have observed that more than 28 % data are missing .Among 8519 entries,there are total 2410 data are missing.")
            
            st.markdown("### BEFORE IMPUTATION :")
            col1,col2=st.columns(2)
            with col1:
                st.markdown("**COUNTPLOT BEFORE IMPUTATION :**")
                st.image("./image/countplot_outlet.png")

            with col2:
                st.markdown("**VALUE COUNT OF EACH CLASS BEFORE IMPUTATION :**")
                st.image('./image/valuecount_outlet.png')

            st.write("So for this case,I have created one extra class name'not_mentioned',and replace it with all the missing values.")
            
            st.markdown("### AFTER IMPUTATION :")
            col1,col2=st.columns(2)
            with col1:
                st.markdown("**COUNTPLOT AFTER IMPUTATION :**")
                st.image("./image/countp_af.png")

            with col2:
                st.markdown("**VALUE COUNT OF AFTER IMPUTATION :**")
                st.image('./image/valuec_af.png')
            st.write("That's all for thisn column.")

# for DISTRIBUTION :
if show_details:
    if res=='NUMERICAL-PLOTTING':
        if res_dis=="DIST-PLOT":
            column=columns
            st.markdown(f"#### DISTRIBUTION OF {column}-COLUMN")
            fig,ax1=plt.subplots()
            sns.distplot(cleantrain[column],ax=ax1,color='red')
            ax1.grid()
            st.pyplot(fig)

        elif res_dis=="SCATTER-PLOT":
            first_col=columns1
            second_cols=columns2
            st.markdown(f"#### SCATTER-PLOT--{first_col}- VS - {second_cols}")
            fig,ax1=plt.subplots()
            ax1.scatter(cleantrain[first_col],cleantrain[second_cols],color='#20FFCC')
            ax1.grid()
            st.pyplot(fig)

        elif res_dis=="BOX-PLOT":
            column=column_box
            st.markdown(f"#### BOX-PLOT OF {column}")
            fig,ax1=plt.subplots()
            sns.boxplot(cleantrain[column],ax=ax1)
            st.pyplot(fig)


# for categorical columns:
if show_details:
    if res=='CATEGORICAL-PLOTTING':
        if res_dis=="COUNT-PLOT":
            column=columns
            categories=cleaned_train[column].unique()
            st.markdown(f"#### COUNT-PLOT OF {column}")
            fig,ax1=plt.subplots()
            sns.countplot(data=cleaned_train,x=column,ax=ax1)
            ax1.set_xticklabels(categories,rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        if res_dis=="PIE-CHART":
            column=columns
            categories=cleaned_train[column].unique()
            st.markdown(f"#### PIE-CHART OF {column}")
            fig,ax1=plt.subplots()
            cleaned_train[column].value_counts().plot(kind='pie',autopct='%0.1f%%')
            #ax1.set_xticklabels(categories,rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

#for model-comparison:
if show_details:
    if res=='MODEL-COMPARISON':
        if res_dis=="R2-SCORE":

            col1,col2=st.columns(2)
            with col1:
                st.markdown('**MODEL DATASET OF FINAL R2-SCORE :**')
                temp_df=model_result[['Model_Name','r2_score']]
                st.dataframe(temp_df)

            with col2:
                st.markdown('**BAR PLOT OF FINAL R2-SCORE:**')
                fig,ax1=plt.subplots()
                ax1.set_title('BAR-PLOT OF PERFORMANCE THE MODELS',fontweight='bold')
                ax1.bar(model_result['Model_Name'],model_result['r2_score'])
                ax1.set_xticklabels(model_result['Model_Name'],rotation=45, ha='right')
                
                st.pyplot(fig)
            
            st.write('This is the r2 score (you can say accuracy also) with respect to  models.')
            st.write("we can see here that the GredientBoost-Regressor gives us the highest performance ,that's 87.21%.")
        
        if res_dis=="MEAN-SQUARE-ERROR":
            col1,col2=st.columns(2)
            with col1:
                st.markdown('**MEAN-SQUARED-ERROR OF MODELS :**')
                temp_df=model_result[['Model_Name','mean_sqr_err']]
                st.dataframe(temp_df)

            with col2:
                st.markdown('**VISUALIZATION OF MEAN-SQUARED-ERROR FOR EACH MODEL:**')
                fig,ax1=plt.subplots()
                ax1.set_title('BAR-PLOT OF MEAN-SQUARED-ERROR OF MODELS',fontweight='bold')
                ax1.bar(model_result['Model_Name'],model_result['mean_sqr_err'])
                ax1.set_xticklabels(model_result['Model_Name'],rotation=45, ha='right')
                
                st.pyplot(fig)
            
            st.markdown('**This is the overview of the mean-squared-error  with respect to each model.**')

        if res_dis=="ROOT-MEAN-SQUARE-ERROR":
            col1,col2=st.columns(2)
            with col1:
                st.markdown('**ROOT-MEAN-SQUARED-ERROR OF MODELS :**')
                temp_df=model_result[['Model_Name','RMSE']]
                st.dataframe(temp_df)

            with col2:
                st.markdown('**VISUALIZATION OF ROOT-MEAN-SQUARED-ERROR FOR EACH MODEL:**')
                fig,ax1=plt.subplots()
                ax1.set_title('BAR-PLOT OF ROOT-MEAN-SQUARED-ERROR OF MODELS',fontweight='bold')
                ax1.bar(model_result['Model_Name'],model_result['RMSE'])
                ax1.set_xticklabels(model_result['Model_Name'],rotation=45, ha='right')
                
                st.pyplot(fig)
            
            st.markdown('**This is the overview of the root-mean-squared-error  with respect to each model.**')    

        if res_dis=="MEAN-ABSOLUTE-ERROR":
            col1,col2=st.columns(2)
            with col1:
                st.markdown('**MEAN-ABSOLUTE-ERROR OF MODELS :**')
                temp_df=model_result[['Model_Name','mean_abs_error']]
                st.dataframe(temp_df)

            with col2:
                st.markdown('**VISUALIZATION OF MEAN-ABSOLUTE-ERROR FOR EACH MODEL:**')
                fig,ax1=plt.subplots()
                ax1.set_title('BAR-PLOT OF MEAN-ABSOLUTE-ERROR OF MODELS',fontweight='bold')
                ax1.bar(model_result['Model_Name'],model_result['mean_abs_error'])
                ax1.set_xticklabels(model_result['Model_Name'],rotation=45, ha='right')
                
                st.pyplot(fig)
            
            st.markdown('**This is the overview of the mean-absolute-error  with respect to each model.**') 

        if res_dis=="SCORE-COMPARISON":
            col1,col2=st.columns(2)
            with col1:
                st.markdown('**R2-SCORE BEFORE HYPERPARAMETER TUNING :**')
                temp_df=model_result[['Model_Name','Before_param_r2_scr']]
                st.dataframe(temp_df)

            with col2:
                st.markdown('**R2-SCORE AFTER HYPERPARAMETER TUNING :**')
                temp_df=model_result[['Model_Name','r2_score']]
                st.dataframe(temp_df)

            col3,col4=st.columns(2)
            with col3:
                st.markdown('**VISUALIZATION OF MODEL PERFORMANCE BEFORE HYPERPARAMETER TUNING:**')
                fig,ax1=plt.subplots()
                ax1.set_title('BAR-PLOT OF MODEL PERFORMANCE BEFORE HYPERPARAMETER TUNING :',fontweight='bold')
                ax1.bar(model_result['Model_Name'],model_result['Before_param_r2_scr'],color='red')
                ax1.set_xticklabels(model_result['Model_Name'],rotation=45, ha='right')
                ax1.grid()
                st.pyplot(fig)

                

            with col4:
                st.markdown('**VISUALIZATION OF MODEL PERFORMANCE AFTER HYPERPARAMETER TUNING:**')
                fig,ax1=plt.subplots()
                ax1.set_title('BAR-PLOT OF MODEL PERFORMANCE AFTER HYPERPARAMETER TUNING:',fontweight='bold')
                ax1.bar(model_result['Model_Name'],model_result['r2_score'],color='black')
                ax1.set_xticklabels(model_result['Model_Name'],rotation=45, ha='right')
                ax1.grid()
                st.pyplot(fig)

            st.markdown('### DIFFERENCE OF THE SCORE :')
            col1,col2=st.columns(2)
            with col1:
                model_result['score_diff']=model_result['r2_score']-model_result['Before_param_r2_scr']
                temp_df=model_result[['Model_Name','score_diff']]
                st.dataframe(temp_df)

            with col2:
                st.markdown('**VISUALIZATION OF SCORE-DIFFERENCE:**')
                fig,ax1=plt.subplots()
                ax1.set_title('VISUALIZATION OF SCORE-DIFFERENCE:',fontweight='bold')
                ax1.bar(model_result['Model_Name'],model_result['r2_score'],color='black',label='after')
                ax1.bar(model_result['Model_Name'],model_result['Before_param_r2_scr'],width=0.5,color='blue',label='before')
                ax1.set_xticklabels(model_result['Model_Name'],rotation=45, ha='right')
                ax1.legend()
                ax1.grid()
                st.pyplot(fig)

            st.write('Here we can see that GredientBoost-regressor perform the best with 87% accuracy. After tuning the hyper-parameter the r2-score got increased +3% for it .')
                
                
            
                  











