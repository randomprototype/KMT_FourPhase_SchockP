import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from scipy.integrate import quad #Single integral
from scipy.integrate import dblquad
from PIL import Image

def KD_KT(K,delta,T):
    #########Definitions#######################################################
    def f01(x):#weibull densidade (componente fraco)
        return (b1/a1)*((x/a1)**(b1-1))*np.exp(-(x/a1)**b1)
    def f02(x):#weibull densidade (componente forte)
        return (b2/a2)*((x/a2)**(b2-1))*np.exp(-(x/a2)**b2)
    def fx(x):
        return (p*f01(x))+((1-p)*f02(x))
    def fh(h):
        return l*np.exp(-l*h)
    def Fx(x):
        return (p*(1-np.exp(-(x/a1)**b1)))+((1-p)*(1-np.exp(-(x/a2)**b2)))
    def Rx(x):
        return 1-Fx(x)
    def Fh(h):
        return 1-np.exp(-l*h)
    def Rh(h):
        return np.exp(-l*h)
    #####Scenarios#############################################################
    #####Failure between inspections###########################################
    def C1(K,delta,T):
        PROB1=0
        EC1=0
        EL1=0
        for i in range(0, K):
            PROB1=PROB1+(((1-alfa)**i)*(dblquad(lambda h, x: fx(x)*fh(h), delta[i], (delta[i+1]),0,lambda x:(delta[i+1])-x)[0]))
            EL1=EL1+(((1-alfa)**i)*(dblquad(lambda h, x: (x+h)*fx(x)*fh(h), delta[i], (delta[i+1]),0,lambda x:(delta[i+1])-x)[0]))
            EC1=EC1+(((i*ci)+cf)*(((1-alfa)**i)*(dblquad(lambda h, x: fx(x)*fh(h), delta[i], (delta[i+1]),0,lambda x:(delta[i+1])-x)[0]))) + (((1-alfa)**i)*(dblquad(lambda h, x: cd*h*fx(x)*fh(h), delta[i], (delta[i+1]),0,lambda x:(delta[i+1])-x)[0]))
        return PROB1, EC1, EL1
    ####Replacement at inspection##############################################
    def C2(K,delta,T):
        PROB2=0
        EC2=0
        EL2=0
        for i in range(0, K):
            PROB2=PROB2+(((1-alfa)**i)*(1-beta)*(quad(lambda x: fx(x)*(1-Fh((delta[i+1])-x)),delta[i], (delta[i+1]))[0]))
            EC2=EC2+((((i+1)*ci)+cr)*(((1-alfa)**i)*(1-beta)*(quad(lambda x: fx(x)*(1-Fh((delta[i+1])-x)),delta[i], (delta[i+1]))[0]))) + (((1-alfa)**i)*(1-beta)*(quad(lambda x: cd*(delta[i+1]-x)*fx(x)*(1-Fh((delta[i+1])-x)),delta[i], (delta[i+1]))[0]))
            EL2=EL2+((delta[i+1])*(((1-alfa)**i)*(1-beta)*(quad(lambda x: fx(x)*(1-Fh((delta[i+1])-x)),delta[i], (delta[i+1]))[0])))
        return PROB2, EC2, EL2
    ####Failure after all inspections and before T#############################
    def C3(K,delta,T):
        PROB3=((1-alfa)**(K))*(dblquad(lambda h, x: fx(x)*fh(h), delta[K], T,0,lambda x:T-x)[0])
        EC3=(((K*ci)+cf)*PROB3) + (((1-alfa)**(K))*(dblquad(lambda h, x: cd*h*fx(x)*fh(h), delta[K], T,0,lambda x:T-x)[0]))
        EL3=((1-alfa)**(K))*(dblquad(lambda h, x: (x+h)*fx(x)*fh(h), delta[K], T,0,lambda x:T-x)[0])
        return PROB3, EC3, EL3
    ####Replacement at T#######################################################
    def C4(K,delta,T):
        PROB4=((1-alfa)**(K))*quad(lambda x: fx(x)*(1-Fh(T-x)),delta[K],T)[0]
        EC4=(((K*ci)+cr)*PROB4) + (((1-alfa)**(K))*quad(lambda x: cd*(T-x)*fx(x)*(1-Fh(T-x)),delta[K],T)[0])
        EL4=T*PROB4
        return PROB4, EC4, EL4
    ####Replacement at T without defect########################################
    def C5(K,delta,T):
        PROB5=((1-alfa)**(K))*Rx(T)
        EC5=((K*ci)+cr)*PROB5
        EL5=T*PROB5
        return PROB5, EC5, EL5
    ####Failure after some false negatives#####################################
    def C6(K,delta,T):
        PROB6=0
        EC6=0
        EL6=0
        for i in range(0, K-1):
            for j in range(i+1, K):
                PROB6=PROB6+(((1-alfa)**i)*(beta**(j-i))*(dblquad(lambda h, x: fx(x)*fh(h), delta[i], (delta[i+1]),lambda x:(delta[j])-x,lambda x:(delta[j+1])-x)[0]))
                EL6=EL6+(((1-alfa)**i)*(beta**(j-i))*(dblquad(lambda h, x: (x+h)*fx(x)*fh(h), delta[i], (delta[i+1]),lambda x:(delta[j])-x,lambda x:(delta[j+1])-x)[0]))
                EC6=EC6+(((j*ci)+cf)*(((1-alfa)**i)*(beta**(j-i))*(dblquad(lambda h, x: fx(x)*fh(h), delta[i], (delta[i+1]),lambda x:(delta[j])-x,lambda x:(delta[j+1])-x)[0]))) + (((1-alfa)**i)*(beta**(j-i))*(dblquad(lambda h, x: cd*h*fx(x)*fh(h), delta[i], (delta[i+1]),lambda x:(delta[j])-x,lambda x:(delta[j+1])-x)[0]))          
        return PROB6, EC6, EL6
    ####Replacement at inspection after some false negativies##################
    def C7(K,delta,T):
        PROB7=0
        EC7=0
        EL7=0
        for i in range(0, K-1):
            for j in range(i+2,K+1):
                PROB7=PROB7+(((1-alfa)**i)*(beta**(j-i-1))*(1-beta)*(quad(lambda x: fx(x)*Rh((delta[j])-x),delta[i], (delta[i+1]))[0]))
                EC7=EC7+(((j*ci)+cr)*(((1-alfa)**i)*(beta**(j-i-1))*(1-beta)*(quad(lambda x: fx(x)*Rh((delta[j])-x),delta[i], (delta[i+1]))[0]))) + (((1-alfa)**i)*(beta**(j-i-1))*(1-beta)*(quad(lambda x: cd*(delta[j]-x)*fx(x)*Rh((delta[j])-x),delta[i], (delta[i+1]))[0]))
                EL7=EL7+((delta[j])*(((1-alfa)**i)*(beta**(j-i-1))*(1-beta)*(quad(lambda x: fx(x)*Rh((delta[j])-x),delta[i], (delta[i+1]))[0])))
        return PROB7, EC7, EL7
    ####Replacement by false positives#########################################
    def C8(K,delta,T):
        PROB8=0
        EC8=0
        EL8=0
        for i in range(0,K):
            PROB8=PROB8+(((1-alfa)**i)*alfa*Rx(delta[i+1]))
            EC8=EC8+(((i+1)*ci)+cr)*(((1-alfa)**i)*alfa*Rx(delta[i+1]))
            EL8=EL8+(delta[i+1]*(((1-alfa)**i)*alfa*Rx(delta[i+1])))
        return PROB8, EC8, EL8
    ####Failure after sucessive false negatives after inspections##############
    def C9(K,delta,T):
        PROB9=0
        EC9=0
        EL9=0
        for i in range(0,K):
            PROB9=PROB9+((((1-alfa)**i)*(beta**(K-i))*(dblquad(lambda h, x: fx(x)*fh(h), delta[i], delta[i+1],lambda x: delta[K]-x, lambda x:T-x)[0])))
            EC9=EC9+((K*ci)+cf)*((((1-alfa)**i)*(beta**(K-i))*(dblquad(lambda h, x: fx(x)*fh(h), delta[i], delta[i+1],lambda x:delta[K]-x, lambda x:T-x)[0]))) + ((((1-alfa)**i)*(beta**(K-i))*(dblquad(lambda h, x: cd*h*fx(x)*fh(h), delta[i], delta[i+1],lambda x:delta[K]-x, lambda x:T-x)[0])))
            EL9=EL9+((((1-alfa)**i)*(beta**(K-i))*(dblquad(lambda h, x: (x+h)*fx(x)*fh(h), delta[i], delta[i+1],lambda x:delta[K]-x, lambda x:T-x)[0])))
        return PROB9, EC9, EL9
    #####Replacement at T after sucessive false negatives######################
    def C10(K,delta,T):
        PROB10=0
        EC10=0
        EL10=0
        for i in range(0,K):
            PROB10=PROB10+(((1-alfa)**i)*(beta**(K-i))*(quad(lambda x: fx(x)*Rh(T-x),delta[i], (delta[i+1]))[0]))
            EC10=EC10+((K*ci)+cr)*(((1-alfa)**i)*(beta**(K-i))*(quad(lambda x: fx(x)*Rh(T-x),delta[i], (delta[i+1]))[0])) + (((1-alfa)**i)*(beta**(K-i))*(quad(lambda x: cd*(T-x)*fx(x)*Rh(T-x),delta[i], (delta[i+1]))[0]))
            EL10=EL10+T*(((1-alfa)**i)*(beta**(K-i))*(quad(lambda x: fx(x)*Rh(T-x),delta[i], (delta[i+1]))[0]))
        return PROB10, EC10, EL10

    C1=C1(K,delta,T)
    C2=C2(K,delta,T)
    C3=C3(K,delta,T)
    C4=C4(K,delta,T)
    C5=C5(K,delta,T)
    C6=C6(K,delta,T)
    C7=C7(K,delta,T)
    C8=C8(K,delta,T)
    C9=C9(K,delta,T)
    C10=C10(K,delta,T)

    TOTAL_EC=C1[1]+C2[1]+C3[1]+C4[1]+C5[1]+C6[1]+C7[1]+C8[1]+C9[1]+C10[1]
    TOTAL_EL=C1[2]+C2[2]+C3[2]+C4[2]+C5[2]+C6[2]+C7[2]+C8[2]+C9[2]+C10[2]
    return TOTAL_EC/TOTAL_EL

def main():
    #criando 3 colunas
    col1, col2, col3= st.columns(3)
    foto = Image.open('randomen.png')
    #st.sidebar.image("randomen.png", use_column_width=True)
    #inserindo na coluna 2
    col2.image(foto, use_column_width=True)
    #O código abaixo centraliza e atribui cor
    st.markdown("<h2 style='text-align: center; color: #306754;'>HyPAIRS - Hybrid Policy of Aperiodic Inspections and Replacement System</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #F3F3F3; padding: 10px; text-align: center;">
          <p style="font-size: 20px; font-weight: bold;">An aperiodic inspection and replacement policy based on the delay-time model with component-lifetime heterogeneity</p>
          <p style="font-size: 15px;">By: Victor H. R. Lima, Rafael, G. N. Paiva, Augusto J. S. Rodrigues, Hanser S. J. González & Cristiano A. V. Cavalcante</p>
        </div>
        """, unsafe_allow_html=True)

    menu = ["Cost-rate", "Information", "Website"]
    
    choice = st.sidebar.selectbox("Select here", menu)
    
    if choice == menu[0]:
        st.header(menu[0])
        st.subheader("Insert the parameter values below:")
        
        global a2,b2,a1,b1,p,l,alfa,beta,ci,cr,cf,cd
        a2=st.number_input("Insert the scale parameter for the defect arrival distribution of “strong” components (η\u2081)", min_value = 0.0, value = 3.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the defect arrival for the stronger component.")
        b2=st.number_input("Insert the shape parameter for the defect arrival distribution of “strong” components (β\u2082)", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter specifies the shape parameter for the Weibull distribution, representing the defect arrival for the stronger component.")
        a1=st.number_input("Insert the scale parameter for the defect arrival distribution of “weak” components (η\u2081)", min_value = 3.0, value = 18.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the defect arrival for the weaker component.")
        b1=st.number_input("Insert the shape parameter for the defect arrival distribution of “weak” components (β\u2082)", min_value = 1.0, max_value=5.0, value = 5.0, help="This parameter specifies the shape parameter for the Weibull distribution, representing the defect arrival for the weaker component.")
        p=st.number_input("Insert the mixture parameter (p)", min_value = 0.0, max_value=1.0, value = 0.10, help="This parameter indicates the proportion of the weaker component within the total population of components.")
        l=st.number_input("Insert the rate of the exponential distribution for delay-time (λ)", min_value = 0.0, value = 2.0, help="This parameter defines the rate of the Exponential distribution, which governs the transition from the defective to the failed state of a component.")
        alfa=st.number_input("Insert the false-positive probability (\u03B1)", min_value = 0.0, max_value=1.0, value = 0.1, help="This parameter represents the probability of indicating a defect during inspection when, in fact, it does not exist.")
        beta=st.number_input("Insert the false-negative probability (\u03B5)", min_value = 0.0, max_value=1.0, value = 0.15, help="This parameter represents the probability of not indicating a defect during inspection when, in fact, it does exist.")
        ci=st.number_input("Insert cost of inspection (C_{I})", min_value = 0.0, value = 0.05, help="This parameter represents the cost of conducing an inspection.")
        cr=st.number_input("Insert cost of replacement (inspections and age-based) (C_{R})", min_value = 0.0, value = 1.0, help="This parameter represents the cost associated with preventive replacements, whether performed during inspections or when the age-based threshold is reached.")
        cf=st.number_input("Insert cost of failure (C_{F})", min_value = 0.0, value = 10.0, help="This parameter represents the replacement cost incurred when a component fails.")
        cd=st.number_input("Insert cost of defective by time unit (C_{D})", min_value = 0.0, value = 0.01, help="This parameter represents the unitary cost associated with the time in which the component stays in defective state for each time unit.")
        
        col1, col2 = st.columns(2)
        
        Delta=[0]
        st.subheader("Insert the variable values below:")
        K=int(st.text_input("Insert the number of inspections (K)", value=4))
        if (K<0):
            K=0
        Value=2
        if (K>0):
            for i, col in enumerate(st.columns(K)):
                col.write(f"**{i+1}-th inspection:**")
                Delta.append(col.number_input("Insp. Mom. (Δ)", value=Value*(i+1), min_value=Delta[i-1], key=f"Delta_{i}"))
        T = st.number_input("Insert the moment for the age-based preventive action (T)",value=(K+1)*Value,min_value=Delta[-1])
        
        st.subheader("Click on botton below to run this application:")    
        botao = st.button("Get cost-rate")
        if botao:
            st.write("---RESULT---")
            st.write("Cost-rate", KD_KT(K, Delta, T))
         
    if choice == menu[1]:
        st.header(menu[1])
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>This app is dedicated to compute the cost-rate for a hybrid aperiodic inspection and age-based maintenance policy. We assume a single system operating under Delay-Time Modeling (DTM) with a heterogeneous component lifetime, each having distinct defect arrival distributions. Component renovation occurs either after a failure (corrective maintenance) or during inspections, once a defect is detected or if the age-based threshold is reached (preventive maintenance). We considered false-positive and false-negative probabilities during the inspection.</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>The app computes the cost-rate for a specific solution—defined by the number of inspections (K), inspection moments (Δ) and the age-based threshold (T).</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>For further questions or information on finding the optimal solution, please contact one of the email addresses below.</h6>", unsafe_allow_html=True)
        
        st.write('''

v.h.r.lima@random.org.br

r.g.n.paiva@random.org.br

a.j.s.rodrigues@random.org.br

h.s.j.gonzalez@random.org.br

c.a.v.cavalcante@random.org.br

''' .format(chr(948), chr(948), chr(948), chr(948), chr(948)))       
    if choice==menu[2]:
        st.header(menu[2])
        
        st.write('''The Research Group on Risk and Decision Analysis in Operations and Maintenance was created in 2012 
                 in order to bring together different researchers who work in the following areas: risk, maintenance a
                 nd operation modelling. Learn more about it through our website.''')
        st.markdown('[Click here to be redirected to our website](https://sites.ufpe.br/random/#page-top)',False)        
if st._is_running_with_streamlit:
    main()
else:
    sys.argv = ["streamlit", "run", sys.argv[0]]
    sys.exit(stcli.main())
