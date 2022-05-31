#!/usr/bin/env python
# coding: utf-8

# # Analyze A/B Test Results 
# 
# This project explores concepts in probability, hypothesis testing, and regression. The document is organised into the following sections: 
# 
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# - [Final Check](#finalcheck)
# - [Submission](#submission)
# 
# Specific programming tasks are marked with a **ToDo** tag. 
# 
# <a id='intro'></a>
# ## Introduction
# 
# The goal of this project is to understand the results of an A/B test run by an e-commerce website to determine if the company should:
# - Implement the new webpage, 
# - Keep the old webpage, or 
# - Perhaps run the experiment longer to make their decision.
# 
# 
# Below is the description of the data, there are a total of 5 columns:
# 
# <center>
# 
# |Data columns|Purpose|Valid values|
# | ------------- |:-------------| -----:|
# |user_id|Unique ID|Int64 values|
# |timestamp|Time stamp when the user visited the webpage|-|
# |group|In the current A/B experiment, the users are categorized into two broad groups. <br>The `control` group users are expected to be served with `old_page`; and `treatment` group users are matched with the `new_page`. <br>However, **some inaccurate rows** are present in the initial data, such as a `control` group user is matched with a `new_page`. |`['control', 'treatment']`|
# |landing_page|It denotes whether the user visited the old or new webpage.|`['old_page', 'new_page']`|
# |converted|It denotes whether the user decided to pay for the company's product. Here, `1` means yes, the user bought the product.|`[0, 1]`|
# </center>
# 
# 

# 
# <a id='probability'></a>
# ## Part I - Probability
# 

# In[1]:


# importing relevant libraries

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# ### ToDo 1.1
# **a.** Read in the `ab_data.csv` data. Store it in `df`. 
# Use your dataframe to answer the questions in Quiz 1 of the classroom.

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# **b.** Use the cell below to find the number of rows in the dataset.

# In[3]:


len(df)


# In[4]:


df.tail()


# **c.** The number of unique users in the dataset.

# In[5]:


df.nunique()


# In[6]:


df.user_id.nunique()


# **d.** The proportion of users converted.

# In[7]:


df.converted.sum()/290584


# **e.** The number of times when the "group" is `treatment` but "landing_page" is not a `new_page`.

# In[8]:


treat_old = df.query("group == 'treatment' and landing_page == 'old_page'")
treat_new = df.query("group == 'control' and landing_page == 'new_page'")

len(treat_old) + len(treat_new)
    
    


# **f.** Do any of the rows have missing values?

# In[9]:


df.info()


# ### ToDo 1.2  
# In a particular row, the **group** and **landing_page** columns should have either of the following acceptable values:
# 
# |user_id| timestamp|group|landing_page|converted|
# |---|---|---|---|---|
# |XXXX|XXXX|`control`| `old_page`|X |
# |XXXX|XXXX|`treatment`|`new_page`|X |
# 
# 
# It means, the `control` group users should match with `old_page`; and `treatment` group users should matched with the `new_page`. This is not true for all the users currently.
# 
# For the rows where `treatment` does not match with `new_page` or `control` does not match with `old_page`, we cannot be sure if such rows truly received the new or old wepage. Therefore, the next action will be to remove the inaccurate rows. 
# 
# 
# 
# **a.** Create a new dataset doesn't contain these inaqurate rows.  Store your new dataframe in **df2**.

# In[10]:


# Remove the inaccurate rows, and store the result in a new dataframe df2
df2 = df.query("group == 'treatment' and landing_page == 'new_page'")
df2 = df2.append(df.query("group == 'control' and landing_page == 'old_page'"))


# In[11]:


# Double Check all of the incorrect rows were removed from df2 - 
# Output of the statement below should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# ### ToDo 1.3  
# 

# **a.** How many unique **user_id**s are in **df2**?

# In[12]:


df2.user_id.nunique()


# **b.** There is one **user_id** repeated in **df2**.  What is it?

# In[13]:


df2[df2['user_id'].duplicated()]


# **c.** Display the rows for the duplicate **user_id**? 

# In[14]:


df2[df2['user_id'] == 773192]


# **d.** Remove **one** of the rows with a duplicate **user_id**, from the **df2** dataframe.

# In[15]:


# Remove one of the rows with a duplicate user_id..
 
df2 = df2.drop(2893)

# Check again if the row with a duplicate user_id is deleted or not
df2[df2['user_id'] == 773192]


# In[16]:


df2.head()


# ### ToDo 1.4  
# Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# **a.** What is the probability of an individual converting regardless of the page they receive?<br><br>
# 
# >**Tip**: The probability  you'll compute represents the overall "converted" success rate in the population and you may call it $p_{population}$.
# 
# 

# In[17]:


df2.converted.mean()


# **b.** Given that an individual was in the `control` group, what is the probability they converted?

# In[18]:


len(df2.query("group == 'control' and converted == '1'"))/len(df2.query("group == 'control'"))

#basically, x in control and converted (divide by)
# x in control


# In[19]:


#Or we could do it like this:

control_prob = df2.query("group == 'control'")['converted'].mean()
control_prob


# **c.** Given that an individual was in the `treatment` group, what is the probability they converted?

# In[20]:


treatment_prob = df2.query("group == 'treatment'")['converted'].mean()
treatment_prob


# >**Tip**: The probabilities you've computed in the points (b). and (c). above can also be treated as conversion rate. 
# Calculate the actual difference  (`obs_diff`) between the conversion rates for the two groups. You will need that later.  

# In[21]:


# Calculate the actual difference (obs_diff) between the conversion rates for the two groups.

obs_diff = control_prob - treatment_prob
obs_diff


# **d.** What is the probability that an individual received the new page?

# In[22]:


df2.query('landing_page == "new_page"').shape[0]/df2.shape[0]


# In[23]:


#or we could do it like this
len(df2.query('landing_page == "new_page"')) / len(df2)


# **e.** Consider your results from parts (a) through (d) above, and explain below whether the new `treatment` group users lead to more conversions.

# >**Based on the probabilities of `treatment_prob` and `control_prob`, I can say that there isn't much difference between whether the treatment group led to more conversions than the control group. The control group has slightly more conversions, but this difference seems to be negligible**

# <a id='ab_test'></a>
# ## Part II - A/B Test
# 
# Since a timestamp is associated with each event, you could run a hypothesis test continuously as long as you observe the events. 
# 
# However, then the hard questions would be: 
# - Do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  
# - How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# ### ToDo 2.1
# 
# 
# > Recall that you just calculated that the "converted" probability (or rate) for the old page is *slightly* higher than that of the new page (ToDo 1.4.c). 
# 
# What should be your null and alternative hypotheses (

# >
# >**$H_0$** = **$p_{old}$** = **$p_{new}$**
# 
# >**$H_1$** = **$p_{new}$** > **$p_{old}$**
# >
# 

# ### ToDo 2.2 - Null Hypothesis $H_0$ Testing
# Under the null hypothesis $H_0$, assume that $p_{new}$ and $p_{old}$ are equal. Furthermore, assume that $p_{new}$ and $p_{old}$ both are equal to the **converted** success rate in the `df2` data regardless of the page. So, our assumption is: <br><br>
# <center>
# $p_{new}$ = $p_{old}$ = $p_{population}$
# </center>
# 
# In this section, I will: 
# 
# - Simulate (bootstrap) sample data set for both groups, and compute the  "converted" probability $p$ for those samples. 
# 
# 
# - Use a sample size for each group equal to the ones in the `df2` data.
# 
# 
# - Compute the difference in the "converted" probability for the two samples above. 
# 
# 
# - Perform the sampling distribution for the "difference in the converted probability" between the two simulated-samples over 10,000 iterations; and calculate an estimate. 
# 
# 
# 
# 

# **a.** What is the **conversion rate** for $p_{new}$ under the null hypothesis? 

# In[24]:


p_new = df2.converted.mean()
p_new


# **b.** What is the **conversion rate** for $p_{old}$ under the null hypothesis? 

# In[25]:


# remember, the assumption is that p_new = p_old
p_old = df2.converted.mean()


# **c.** What is $n_{new}$, the number of individuals in the treatment group? <br><br>
# *Hint*: The treatment group users are shown the new page.

# In[26]:


n_new = df2.query('landing_page == "new_page"').shape[0]
n_new


# **d.** What is $n_{old}$, the number of individuals in the control group?

# In[27]:


n_old = df2.query('group == "control"').shape[0]
n_old


# **e. Simulate Sample for the `treatment` Group**<br> 
# Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null hypothesis.  <br><br>
# 
# 

# In[28]:


# Simulate a Sample for the treatment Group
new_page_converted = np.random.choice([0, 1], n_new, p = [p_new, 1-p_new])
new_page_converted


# **f. Simulate Sample for the `control` Group** <br>
# Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null hypothesis. <br> Store these $n_{old}$ 1's and 0's in the `old_page_converted` numpy array.

# In[29]:


# Simulate a Sample for the control Group
old_page_converted = np.random.choice([0, 1], n_old, p = [p_old, 1-p_old])


# **g.** Find the difference in the "converted" probability $(p{'}_{new}$ - $p{'}_{old})$ for your simulated samples from the parts (e) and (f) above. 

# In[30]:


print(new_page_converted.mean() - old_page_converted.mean())


# 
# **h. Sampling distribution** <br>
# Re-create `new_page_converted` and `old_page_converted` and find the $(p{'}_{new}$ - $p{'}_{old})$ value 10,000 times using the same simulation process you used in parts (a) through (g) above. 
# 
# <br>
# Store all  $(p{'}_{new}$ - $p{'}_{old})$  values in a NumPy array called `p_diffs`.

# In[31]:


p_diffs = []
for i in range(10000):
    
    # 1st parameter dictates the choices you want.  In this case [1, 0]
    p_new_10k = np.random.choice([1, 0],n_new,replace = True,p = [p_new, 1-p_new])
    p_old_10k = np.random.choice([1, 0],n_old,replace = True,p = [p_old, 1-p_old])
    p_new_10k = p_new_10k.mean()
    p_old_10k = p_old_10k.mean()
    p_diffs.append(p_new_10k-p_old_10k)


# In[32]:


p_diffs = np.array(p_diffs)


# In[33]:


len(p_diffs)


# **i. Histogram**<br> 
# Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.<br><br>

# In[34]:


#plt.hist(p_diffs);
plt.hist(p_diffs)
plt.title('Graph of p_diffs')#title of graphs
plt.xlabel('Page difference') # x-label of graphs
plt.ylabel('Count') # y-label of graphs


# In[35]:


obs_diff = treatment_prob - control_prob

low_prob = (p_diffs < obs_diff).mean()
high_prob = (p_diffs.mean() + (p_diffs.mean() - obs_diff) < p_diffs).mean()

plt.hist(p_diffs);
plt.axvline(obs_diff, color='red');
#plt.axvline(p_diffs.mean() + (p_diffs.mean() - obs_diff), color='red');

p_val = low_prob + high_prob
print(p_val)


# **j.** What proportion of the **p_diffs** are greater than the actual difference observed in the `df2` data?

# In[36]:


(p_diffs > obs_diff).mean()


# **k.** Please explain in words what you have just computed in part **j** above.  
#  - What is this value called in scientific studies?  
#  - What does this value signify in terms of whether or not there is a difference between the new and old pages? 

# >**The value calculated is the p-value. The p-value here is about `0.9` and this is higher than the type error rate of `0.05`. This p-value tells us that about `90%` of the population is highter than the actual difference observed. When a p-value is less than `0.05`, we are to reject the null hypothesis, because that is a statistically significant result. In this case, the p-value is higher than `0.05`, and as such, we cannot reject the null hypothesis that says that `p_new = p_old`. This means that we cannot say that the new page is better than the old page.**

# 
# 
# **l. Using Built-in Methods for Hypothesis Testing**<br>
# We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. 
# 
# Fill in the statements below to calculate the:
# - `convert_old`: number of conversions with the old_page
# - `convert_new`: number of conversions with the new_page
# - `n_old`: number of individuals who were shown the old_page
# - `n_new`: number of individuals who were shown the new_page
# 

# In[37]:





# number of conversions with the old_page
convert_old = df2.query('converted == 1 and landing_page == "old_page"').shape[0]

# number of conversions with the new_page
convert_new = df2.query('converted == 1 and landing_page == "new_page"').shape[0]

# number of individuals who were shown the old_page
n_old = len(df2.query('landing_page == "old_page"'))

# number of individuals who received new_page
n_new = len(df2.query('landing_page == "new_page"'))


# 
# 
# ---
# ### About the two-sample z-test
# 
# 
# Next step is to make a decision to reject or fail to reject the null hypothesis based on comparing these two values: 
# - $Z_{score}$
# - $Z_{\alpha}$ or $Z_{0.05}$, also known as critical value at 95% confidence interval.  $Z_{0.05}$ is 1.645 for one-tailed tests,  and 1.960 for two-tailed test. You can determine the $Z_{\alpha}$ from the z-table manually. 
# 
# Decide if your hypothesis is either a two-tailed, left-tailed, or right-tailed test. Accordingly, reject OR fail to reject the  null based on the comparison between $Z_{score}$ and $Z_{\alpha}$. 
# 
# 
# 
# 

# In[38]:


import statsmodels.api as sm
# ToDo: Complete the sm.stats.proportions_ztest() method arguments
z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
print(z_score, p_value)


# **n.** What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?<br><br>
# 

# > - The z-score here is about `1.31` and this is a positive value. 
# - In a right-tailed test, we reject null if $Z_{score}$ > $Z_{\alpha}$ or less than -($Z_{\alpha}$). The $Z_{\alpha}$ (the critical value at 95% confidence interval) is `1.645` for a one-tailed test. 
# - In this test, $Z_{score}$ < $Z_{\alpha}$ (`1.31` < `1.645`) and as such, we fail to reject the null hypothesis.
# - With a positive z-score, we can also deduce that the z-score 1.31 standard deviations higher than the mean. 
# - The p-value obtained here is also similar to the p-value obtained before, i.e., ~`0.9`, and as such, we fail to reject the null hypothesis that says that `p_new = p_old`. This means that we cannot say that the new page is better than the old page. 

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# ### ToDo 3.1 
# In this final part, I will show that the result achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# **a.** Since each row in the `df2` data is either a conversion or no conversion, what type of regression should you be performing in this case?

# >**logistic regression.**

# **b.** The goal is to use **statsmodels** library to fit the regression model you specified in part **a.** above to see if there is a significant difference in conversion based on the page-type a customer receives. However, you first need to create the following two columns in the `df2` dataframe:
#  1. `intercept` - It should be `1` in the entire column. 
#  2. `ab_page` - It's a dummy variable column, having a value `1` when an individual receives the **treatment**, otherwise `0`.  

# In[39]:


df2.head()


# In[40]:


df_logreg = df2.copy()
df_logreg.head()


# In[41]:


# addding an intercept
df_logreg['intercept'] = 1

# creating dummy columns
df_logreg['ab_page'] = pd.get_dummies(df_logreg['group'])['treatment']


# checking again
df_logreg.head()


# **c.** Use **statsmodels** to instantiate your regression model on the two columns you created in part (b). above, then fit the model to predict whether or not an individual converts. 
# 

# In[42]:


log_mod =sm.Logit(df_logreg['converted'], df_logreg[['intercept', 'ab_page']])
results=log_mod.fit() 


# **d.** Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[43]:


results.summary2()


# **e.** What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br> 
# 

# >**The p-value associated with ab_page is 0.1899.**
# **The null and alternative hypothesis associated with the regression model are as follows:**<br>
# 
# 
# >**$H_0$** = **$p_{old}$** - **$p_{new}$** = 0 <br>
# **$H_1$** = **$p_{new}$** - **$p_{old}$** ≠ 0 <br>
# 
# >In part II, the test was a one sided (right tailed test), while the logistic regression approach is a two tailed test. Inspite of their differences, the summary table above shows the same result for the `ab_page` z-test, i.e., `1.3109`
# Another thing to note is that the p-value for ab_page is 0.1899, and this is higher than the 0.05, meaning that we can fail to reject the null hypothesis in this case. This means that we cannot say that the new page is better than the old page.
# >

# **f.** Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# In[44]:


# duration of the experiment
duration = np.array(pd.to_datetime(df.timestamp).sort_values(ascending=True))
td = duration[-1] - duration[0]
days = td.astype('timedelta64[D]')
days / np.timedelta64(1, 'D')


# > - The duration of the project may influence the results seen so far. 21 days (as evaluated above) can be considered a short time to run the A/B tests, as it may allow change aversion effects to occur.
# - The countries where the page is opened may also affect individual converts where language barrier is a challenge.

# **g. Adding countries**<br> 
# Now along with testing if the conversion rate changes for different pages, I'll also add an effect based on which country a user lives in. 
# 
# 1. I will read in the **countries.csv** dataset and merge together your `df2` datasets on the appropriate rows. 
# 
# 2. Does it appear that country had an impact on conversion?  To answer this question, I'll consider the three unique values, `['UK', 'US', 'CA']`, in the `country` column. Create dummy variables for these country columns. 
# 

# In[45]:


# Read the countries.csv
df_countries = pd.read_csv('countries.csv')
df_countries.head()


# In[46]:


# Join with the df2 dataframe
df_merged = df2.join(df_countries.set_index('user_id'), on='user_id')
# checking...
df_merged.head()


# In[47]:


# Create the necessary dummy variables
df_merged[['canada','uk','us']] = pd.get_dummies(df_merged['country'])

df_merged.head()


# In[48]:


# Using Canada as baseline, we'll drop Canada..
df_merged.drop(['canada'], axis=1, inplace=True)


# **h. Fit your model and obtain the results**<br> 
# 
# >**Tip**: Conclusions should include both statistical reasoning, and practical reasoning for the situation. 
# 
# >**Hints**: 
# - Look at all of p-values in the summary, and compare against the Type I error rate (0.05). 
# - Can you reject/fail to reject the null hypotheses (regression model)?
# - Comment on the effect of page and country to predict the conversion.
# 

# In[49]:


# Fit your model

# creating intercept
df_merged['intercept'] = 1

logit_mod = sm.Logit(df_merged['converted'], df_merged[['intercept','us','uk']])
results = logit_mod.fit()


# In[50]:


# and summarize the results

results.summary2()


# In[51]:


# Calculating the exponential of the results for better interpretation

1/np.exp(0.0408), np.exp(0.0507)


# >**Results Summary.**
# Based on the results above, the people in the US are 0.96 times more likely to convert compared to users in Canada. Also, the people in the UK are 0.507 times more likely to convert compared to users in Canada. This result however may nor be statistically significant as both the US and the UK have produced p-values higher than 0.05. This is similar to the results from the initial p-value, and z-score where we filed to reject the null hypothesis

# # Conclusion
# 
# The aim of this project was to understand and determine whether a company should implement a new page or keep the old one. To achieve this we used:
# - Probability test.
# - A/B test.
# - Regression.
# 
# ### Probability Test
# - With the probability test, we found out that the probability that an individual received the new page was `0.500`. Meaning that the population had approximately a 50% chance in receiving the new page
# - We also found that there was not mucg difference between the probabilty that an individual in either the treatment group or control group converted.
# 
# ### A/B Test
# - In the A/B Tests, the hypothesis was set up to determine if the new page gives better conversion or not.
# - A/B test results showed a p-value of approximately `0.9` which is higher than the type error rate of `0.05`.
# - With this result, we failed to reject the null hypothesis; **$H_0$** = **$p_{old}$** = **$p_{new}$**.
# - The in-built z-test (right tailed) was also computed. With a z-score of `1.31`, we failed to reject the null hypothesis, just as we did previously with the p-value results.
# 
# ### Regression Test
# - A logistic regression test was conducted to explore two possible outcomes, i.e, to determine if the new page is better or not.
# - The logistic regression results produced the same z-score as the one obtained in the A/B test. The logistic regression also produced a p-value of `0.190`, higher than the type error value of `0.05`, and as such, we again failed to reject the null hypothesis.
# - Countries were considered to have probable impact on conversion rates and as such were added to the model in another logistic regression testing.
# - The results were statistically insignificant and as such, it cannot be said that the countries have any impact on the conversion rate
# 
# ### Considerations
# - The duration of the test was also considered as a probable factor that may impact the results of the test.
# - The duration of the test was determined to be 21 days. 21 days can be considered a relatively short time to run the A/B test.
# - This short time may increase the possibility for change aversion effect to occur, where users may give unfair advantate to the old page.
# 
# 
# ## Resources: 
# -  [Statistics](https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/09%3A_Two-Sample_Problems/9.01%3A_Comparison_of_Two_Population_Means-_Large_Independent_Samples), courtesy www.stats.libretexts.org
# 
# - [Joining Tables](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html) 
# 
# - Investopia: [One Sample Z-Test Example](https://www.investopedia.com/terms/z/z-test.asp#:~:text=If%20the%20value%20of%20z,observed%20average%20of%20the%20samples.)

# <a id='finalcheck'></a>
# ## Final Check!
# 
# Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your notebook to make sure that it satisfies all the specifications mentioned in the rubric. You should also probably remove all of the "Hints" and "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# <a id='submission'></a>
# ## Submission
# You may either submit your notebook through the "SUBMIT PROJECT" button at the bottom of this workspace, or you may work from your local machine and submit on  the last page of this project lesson.  
# 
# 1. Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# 
# 2. Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# 
# 3. Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[52]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




