import streamlit as st
import numpy as np
import sys
from streamlit import cli as stcli
from scipy.integrate import quad #Single integral
from scipy.integrate import dblquad
from PIL import Image
from numba import njit, prange

def Analytics_Cost_rate(K,M,T,n1, b1, mi_menor, n2, b2, mi_maior, l_tx, mi_falha, b, cb, ci, cr, cf, c):
    # Funções preliminares (defeito, delaytime)
    def fx(x):
        return (b1/n1**b1)*(x**(b1-1))*np.exp(-(x/n1)**b1)
    def Fx(x): #weibull acumulada densidade (DEFEITO MENOR)
        return quad(fx, 0, x)[0]
    def Rx(x): #
        return quad(fx, x, np.inf)[0]
    def fy(y):#weibull densidade (DEFEITO MAIOR)
        return (b2/n2**b2)*(y**(b2-1))*np.exp(-(y/n2)**b2)
    def Ry(y): #
        return quad(fy, y, np.inf)[0]
    def fh(h):#exponencial densidade DELAY TIME
        return l_tx*np.exp(-l_tx*h)
    def Fh(h):#exponencial acumulada DELAY TIME
        return 1-np.exp(-l_tx*h)
    def Rh(h):
        return np.exp(-l_tx*h)

    # CENÁRIO 1 - Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO e falha DEGRADAÇÃO chegam entre inspeções menores
    def Scenario1():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M+1):
                P=tplquad(lambda h,y,x:np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,y:0,lambda x,y:(l*M+i)*T-(x+y))[0]
                Cost+=((l*cb+((l*(M-1)+(i-1))*ci)+cf)*P)+tplquad(lambda h,y,x:c*h*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,y:0,lambda x,y:(l*M+i)*T-(x+y))[0]
                Lifetime+=tplquad(lambda h,y,x:(x+h+y)*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,y:0,lambda x,y:(l*M+i)*T-(x+y))[0]
                Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 2 - Defeito menor DEGRADAÇÃO, defeito maior CHOQUE e falha DEGRADAÇÃO chegam entre inspeções menores
    def Scenario2():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M+1):
                P=tplquad(lambda h,w,x:np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,w:0,lambda x,w:(l*M+i)*T-(x+w))[0]
                Cost+=((l*cb+((l*(M-1)+(i-1))*ci)+cf)*P)+tplquad(lambda h,w,x:c*h*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,w:0,lambda x,w:(l*M+i)*T-(x+w))[0]
                Lifetime+=tplquad(lambda h,w,x:(x+h+w)*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,w:0,lambda x,w:(l*M+i)*T-(x+w))[0]
                Probability+=P
        return Cost,Lifetime,Probability

    # CENÁRIO 3 - Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO e falha CHOQUE chegam entre inspeções menores
    def Scenario3():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M+1):
                P=tplquad(lambda q,y,x:np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,y:0,lambda x,y:(l*M+i)*T-(x+y))[0]
                Cost+=((l*cb+((l*(M-1)+(i-1))*ci)+cf)*P)+tplquad(lambda q,y,x:c*q*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,y:0,lambda x,y:(l*M+i)*T-(x+y))[0]
                Lifetime+=tplquad(lambda q,y,x:(x+q+y)*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,y:0,lambda x,y:(l*M+i)*T-(x+y))[0]
                Probability+=P
        return Cost,Lifetime,Probability

    # CENÁRIO 4 - Defeito menor DEGRADAÇÃO, defeito maior CHOQUE e falha CHOQUE chegam entre inspeções menores
    def Scenario4():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M+1):
                P=tplquad(lambda q,w,x:np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,w:0,lambda x,w:(l*M+i)*T-(x+w))[0]
                Cost+=((l*cb+((l*(M-1)+(i-1))*ci)+cf)*P)+tplquad(lambda q,w,x:c*q*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,w:0,lambda x,w:(l*M+i)*T-(x+w))[0]
                Lifetime+=tplquad(lambda q,w,x:(x+q+w)*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x,lambda x,w:0,lambda x,w:(l*M+i)*T-(x+w))[0]
                Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 5 - Defeito menor CHOQUE, defeito maior DEGRADAÇÃO e falha DEGRADAÇÃO chegam entre inspeções menores
    def Scenario5():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M+1):
                P=tplquad(lambda h,y,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,y:0,lambda z,y:(l*M+i)*T-(z+y))[0]
                Cost+=((l*cb+((l*(M-1)+(i-1))*ci)+cf)*P)+tplquad(lambda h,y,z:c*h*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,y:0,lambda z,y:(l*M+i)*T-(z+y))[0]
                Lifetime+=tplquad(lambda h,y,z:(z+h+y)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,y:0,lambda z,y:(l*M+i)*T-(z+y))[0]
                Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 6 - Defeito menor CHOQUE, defeito maior CHOQUE e falha DEGRADAÇÃO chegam entre inspeções menores
    def Scenario6():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M+1):
                P=tplquad(lambda h,w,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,w:0,lambda z,w:(l*M+i)*T-(z+w))[0]
                Cost+=((l*cb+((l*(M-1)+(i-1))*ci)+cf)*P)+tplquad(lambda h,w,z:c*h*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,w:0,lambda z,w:(l*M+i)*T-(z+w))[0]
                Lifetime+=tplquad(lambda h,w,z:(z+h+w)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,w:0,lambda z,w:(l*M+i)*T-(z+w))[0]
                Probability+=P
        return Cost,Lifetime,Probability
    
    
    # CENÁRIO 7 - Defeito menor CHOQUE, defeito maior DEGRADAÇÃO e falha CHOQUE chegam entre inspeções menores
    def Scenario7():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M+1):
                P=tplquad(lambda q,y,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,y:0,lambda z,y:(l*M+i)*T-(z+y))[0]
                Cost+=((l*cb+((l*(M-1)+(i-1))*ci)+cf)*P)+tplquad(lambda q,y,z:c*q*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,y:0,lambda z,y:(l*M+i)*T-(z+y))[0]
                Lifetime+=tplquad(lambda q,y,z:(z+q+y)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,y:0,lambda z,y:(l*M+i)*T-(z+y))[0]
                Probability+=P
        return Cost,Lifetime,Probability
    
    
    # CENÁRIO 8 - Defeito menor CHOQUE, defeito maior CHOQUE e falha CHOQUE chegam entre inspeções menores
    def Scenario8():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M+1):
                P=tplquad(lambda q,w,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,w:0,lambda z,w:(l*M+i)*T-(z+w))[0]
                Cost+=((l*cb+((l*(M-1)+(i-1))*ci)+cf)*P)+tplquad(lambda q,w,z:c*q*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,w:0,lambda z,w:(l*M+i)*T-(z+w))[0]
                Lifetime+=tplquad(lambda q,w,z:(z+q+w)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z,lambda z,w:0,lambda z,w:(l*M+i)*T-(z+w))[0]
                Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 9 - Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO e falha DEGRADAÇÃO chegam entre inspeções menor após um ou mais falso negativo
    def Scenario9():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M+1):
                for i in range(1,j):
                    P=(b**(j-i))*tplquad(lambda h,y,x:np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,y:0,lambda x,y:(l*M+j)*T-(x+y))[0]
                    Cost+=((l*cb+((l*(M-1)+(j-1))*ci)+cf)*P)+(b**(j-i))*tplquad(lambda h,y,x:c*h*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,y:0,lambda x,y:(l*M+j)*T-(x+y))[0]
                    Lifetime+=(b**(j-i))*tplquad(lambda h,y,x:(x+h+y)*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,y:0,lambda x,y:(l*M+j)*T-(x+y))[0]
                    Probability+=P
        return Cost,Lifetime,Probability
    
    
    # CENÁRIO 10 - Defeito menor DEGRADAÇÃO, defeito maior CHOQUE e falha DEGRADAÇÃO chegam entre inspeções menor após um ou mais falso negativo
    def Scenario10():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M+1):
                for i in range(1,j):
                    P=(b**(j-i))*tplquad(lambda h,w,x:np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,w:0,lambda x,w:(l*M+j)*T-(x+w))[0]
                    Cost+=((l*cb+((l*(M-1)+(j-1))*ci)+cf)*P)+(b**(j-i))*tplquad(lambda h,w,x:c*h*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,w:0,lambda x,w:(l*M+j)*T-(x+w))[0]
                    Lifetime+=(b**(j-i))*tplquad(lambda h,w,x:(x+h+w)*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,w:0,lambda x,w:(l*M+j)*T-(x+w))[0]
                    Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 11 - Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO e falha CHOQUE chegam entre inspeções menores após um ou mais falso negativo
    def Scenario11():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M+1):
                for i in range(1,j):
                    P=(b**(j-i))*tplquad(lambda q,y,x:np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,y:0,lambda x,y:(l*M+j)*T-(x+y))[0]
                    Cost+=((l*cb+((l*(M-1)+(j-1))*ci)+cf)*P)+(b**(j-i))*tplquad(lambda q,y,x:c*q*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,y:0,lambda x,y:(l*M+j)*T-(x+y))[0]
                    Lifetime+=(b**(j-i))*tplquad(lambda q,y,x:(x+q+y)*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,y:0,lambda x,y:(l*M+j)*T-(x+y))[0]
                    Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 12 - Defeito menor DEGRADAÇÃO, defeito maior CHOQUE e falha CHOQUE chegam entre inspeções menores após um ou mais falso negativo
    def Scenario12():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M+1):
                for i in range(1,j):
                    P=(b**(j-i))*tplquad(lambda q,w,x:np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,w:0,lambda x,w:(l*M+j)*T-(x+w))[0]
                    Cost+=((l*cb+((l*(M-1)+(j-1))*ci)+cf)*P)+(b**(j-i))*tplquad(lambda q,w,x:c*q*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,w:0,lambda x,w:(l*M+j)*T-(x+w))[0]
                    Lifetime+=(b**(j-i))*tplquad(lambda q,w,x:(x+q+w)*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*np.exp(-l_tx*q),(l*M+(i-1))*T,(l*M+i)*T,lambda x:(l*M+j-1)*T-x,lambda x:(l*M+j)*T-x,lambda x,w:0,lambda x,w:(l*M+j)*T-(x+w))[0]
                    Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 13 - Defeito menor CHOQUE, defeito maior DEGRADAÇÃO e falha DEGRADAÇÃO chegam entre inspeções menores após um ou mais falso negativo
    def Scenario13():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M+1):
                for i in range(1,j):
                    P=(b**(j-i))*tplquad(lambda h,y,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,y:0,lambda z,y:(l*M+j)*T-(z+y))[0]
                    Cost+=((l*cb+((l*(M-1)+(j-1))*ci)+cf)*P)+(b**(j-i))*tplquad(lambda h,y,z:c*h*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,y:0,lambda z,y:(l*M+j)*T-(z+y))[0]
                    Lifetime+=(b**(j-i))*tplquad(lambda h,y,z:(z+h+y)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,y:0,lambda z,y:(l*M+j)*T-(z+y))[0]
                    Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 14 - Defeito menor CHOQUE, defeito maior CHOQUE e falha DEGRADAÇÃO chegam entre inspeções menores após um ou mais falso negativo
    def Scenario14():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M+1):
                for i in range(1,j):
                    P=(b**(j-i))*tplquad(lambda h,w,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,w:0,lambda z,w:(l*M+j)*T-(z+w))[0]
                    Cost+=((l*cb+((l*(M-1)+(j-1))*ci)+cf)*P)+(b**(j-i))*tplquad(lambda h,w,z:c*h*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,w:0,lambda z,w:(l*M+j)*T-(z+w))[0]
                    Lifetime+=(b**(j-i))*tplquad(lambda h,w,z:(z+h+w)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*np.exp(-mi_falha*h)*fh(h),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,w:0,lambda z,w:(l*M+j)*T-(z+w))[0]
                    Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 15 - Defeito menor CHOQUE, defeito maior DEGRADAÇÃO e falha CHOQUE chegam entre inspeções menores após um ou mais falso negativo
    def Scenario15():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M+1):
                for i in range(1,j):
                    P=(b**(j-i))*tplquad(lambda q,y,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,y:0,lambda z,y:(l*M+j)*T-(z+y))[0]
                    Cost+=((l*cb+((l*(M-1)+(j-1))*ci)+cf)*P)+(b**(j-i))*tplquad(lambda q,y,z:c*q*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,y:0,lambda z,y:(l*M+j)*T-(z+y))[0]
                    Lifetime+=(b**(j-i))*tplquad(lambda q,y,z:(z+q+y)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,y:0,lambda z,y:(l*M+j)*T-(z+y))[0]
                    Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 16 - Defeito menor CHOQUE, defeito maior CHOQUE e falha CHOQUE chegam entre inspeções menores após um ou mais falso negativo
    def Scenario16():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M+1):
                for i in range(1,j):
                    P=(b**(j-i))*tplquad(lambda q,w,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,w:0,lambda z,w:(l*M+j)*T-(z+w))[0]
                    Cost+=((l*cb+((l*(M-1)+(j-1))*ci)+cf)*P)+(b**(j-i))*tplquad(lambda q,w,z:c*q*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,w:0,lambda z,w:(l*M+j)*T-(z+w))[0]
                    Lifetime+=(b**(j-i))*tplquad(lambda q,w,z:(z+q+w)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*mi_falha*np.exp(-mi_falha*q)*(np.exp(-l_tx*q)),(l*M+(i-1))*T,(l*M+i)*T,lambda z:(l*M+j-1)*T-z,lambda z:(l*M+j)*T-z,lambda z,w:0,lambda z,w:(l*M+j)*T-(z+w))[0]
                    Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 17 – Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO chega no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor.
    def Scenario17():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M):
                P=dblquad(lambda y,x:np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*Rh((l*M+i)*T-(x+y))*np.exp(-mi_falha*((l*M+i)*T-(x+y))),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x)[0]
                Cost+=((l*cb+((l*(M-1))+i)*ci+cr)*P)+dblquad(lambda y,x:c*((l*M+i)*T-x-y)*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*Rh((l*M+i)*T-(x+y))*np.exp(-mi_falha*((l*M+i)*T-(x+y))),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x)[0]
                Lifetime+=(((l*M)+i)*T)*P
                Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 18 – Defeito menor DEGRADAÇÃO, defeito maior CHOQUE chega no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor.
    def Scenario18():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M):
                P=dblquad(lambda w,x:np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*(np.exp(-(w/n2)**b2))*Rh((l*M+i)*T-(x+w))*np.exp(-mi_falha*((l*M+i)*T-(x+w))),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x)[0]
                Cost+=((l*cb+((l*(M-1))+i)*ci+cr)*P)+dblquad(lambda w,x:c*((l*M+i)*T-x-w)*(np.exp(-mi_menor*x) * fx(x))*(mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l*M+i)*T-(x+w)) * np.exp(-mi_falha*((l*M+i)*T-(x+w))),(l*M+(i-1))*T,(l*M+i)*T,lambda x:0,lambda x:(l*M+i)*T-x)[0]
                Lifetime+=(((l*M)+i)*T)*P
                Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 19 – Defeito menor CHOQUE, defeito maior DEGRADAÇÃO chega no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor.
    def Scenario19():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M):
                P=dblquad(lambda y,z:mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*Rh((l*M+i)*T-(z+y))*np.exp(-mi_falha*((l*M+i)*T-(z+y))),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z)[0]
                Cost+=((l*cb+((l*(M-1))+i)*ci+cr)*P)+dblquad(lambda y,z:c*((l*M+i)*T-z-y)*mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))*np.exp(-mi_maior*y)*fy(y)*Rh((l*M+i)*T-(z+y))*np.exp(-mi_falha*((l*M+i)*T-(z+y))),(l*M+(i-1))*T,(l*M+i)*T,lambda z:0,lambda z:(l*M+i)*T-z)[0]
                Lifetime+=(((l*M)+i)*T)*P
                Probability+=P
        return Cost,Lifetime,Probability
    
    # CENÁRIO 20 – Defeito menor CHOQUE, defeito maior CHOQUE chega no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor.
    def Scenario20():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M):
                  P=(dblquad(lambda w,z:(mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l*M+i)*T-(z+w)) * np.exp(-mi_falha*((l*M+i)*T-(z+w))), (l*M+(i-1))*T, (l*M+i)*T, lambda z: 0, lambda z: (l*M+i)*T-z)[0])
                  Cost+=((l * cb) + (((l*(M-1))+i) * ci)+ cr) * P + (dblquad(lambda w,z:(c*((l*M+i)*T-z-w))*(mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l*M+i)*T-(z+w)) * np.exp(-mi_falha*((l*M+i)*T-(z+w))), (l*M+(i-1))*T, (l*M+i)*T, lambda z: 0, lambda z: (l*M+i)*T-z)[0])
                  Lifetime+=((((l*(M))+i) * T) * P)
                  Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 21 –  Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor APÓS FALSO NEGATIVO.
    def Scenario21():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M):
                for i in range(1,j):
                    P=(b**(j-i)) * (dblquad(lambda y,x: (np.exp(-mi_menor*x) * fx(x)) * (np.exp(-mi_maior*y) * fy(y)) * Rh((l*M+j)*T-(x+y)) * np.exp(-mi_falha*((l*M+j)*T-(x+y))), (l*M+(i-1))*T, (l*M+i)*T, lambda x:  ((l*M+(j-1))*T)-x, lambda x: ((l*M+j)*T)-x)[0])
                    Cost+=(((l * cb) + (((l*(M-1))+j) * ci) + cr) * P) + (b**(j-i)) * (dblquad(lambda y,x: (c*((l*M+j)*T-x-y))*(np.exp(-mi_menor*x) * fx(x)) * (np.exp(-mi_maior*y) * fy(y)) * Rh((l*M+j)*T-(x+y)) * np.exp(-mi_falha*((l*M+j)*T-(x+y))), (l*M+(i-1))*T, (l*M+i)*T, lambda x:  ((l*M+(j-1))*T)-x, lambda x: ((l*M+j)*T)-x)[0])
                    Lifetime+=((((l*(M))+j) * T) * P)
                    Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 22 –  Defeito menor DEGRADAÇÃO, defeito maior CHOQUE no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor APÓS FALSO NEGATIVO.
    def Scenario22():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M):
                for i in range(1,j):
                    P=(b**(j-i)) * (dblquad(lambda w,x: (np.exp(-mi_menor*x) * fx(x)) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l*M+j)*T-(x+w)) * np.exp(-mi_falha*((l*M+j)*T-(x+w))), (l*M+(i-1))*T, (l*M+i)*T, lambda x:  ((l*M+(j-1))*T)-x, lambda x: ((l*M+j)*T)-x)[0])
                    Cost+=(((l * cb) + (((l*(M-1))+j) * ci) + cr) * P) + (b**(j-i)) * (dblquad(lambda w,x: (c*((l*M+j)*T-x-w))*(np.exp(-mi_menor*x) * fx(x)) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l*M+j)*T-(x+w)) * np.exp(-mi_falha*((l*M+j)*T-(x+w))), (l*M+(i-1))*T, (l*M+i)*T, lambda x:  ((l*M+(j-1))*T)-x, lambda x: ((l*M+j)*T)-x)[0])
                    Lifetime+=((((l*(M))+j) * T) * P)
                    Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 23 –  Defeito menor CHOQUE, defeito maior DEGRADAÇÃO no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor APÓS FALSO NEGATIVO.
    def Scenario23():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M):
                for i in range(1,j):
                    P=(b**(j-i)) * (dblquad(lambda y,z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (np.exp(-mi_maior*y) * fy(y)) * Rh((l*M+j)*T-(z+y)) * np.exp(-mi_falha*((l*M+j)*T-(z+y))), (l*M+(i-1))*T, (l*M+i)*T, lambda z:  ((l*M+(j-1))*T)-z, lambda z: ((l*M+j)*T)-z)[0])
                    Cost+=(((l * cb) + (((l*(M-1))+j) * ci) + cr) * P) + (b**(j-i)) * (dblquad(lambda y,z: (c*((l*M+j)*T-z-y))*(mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (np.exp(-mi_maior*y) * fy(y)) * Rh((l*M+j)*T-(z+y)) * np.exp(-mi_falha*((l*M+j)*T-(z+y))), (l*M+(i-1))*T, (l*M+i)*T, lambda z:  ((l*M+(j-1))*T)-z, lambda z: ((l*M+j)*T)-z)[0])
                    Lifetime+=((((l*(M))+j) * T) * P)
                    Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 24 –  Defeito menor CHOQUE, defeito maior CHOQUE no mesmo intervalo de inspeções e substituição preventiva ocorre em inspeção menor APÓS FALSO NEGATIVO.
    def Scenario24():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for j in range(2,M):
                for i in range(1,j):
                    P=(b**(j-i)) * (dblquad(lambda w,z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l*M+j)*T-(z+w)) * np.exp(-mi_falha*((l*M+j)*T-(z+w))), (l*M+(i-1))*T, (l*M+i)*T, lambda z:  ((l*M+(j-1))*T)-z, lambda z: ((l*M+j)*T)-z)[0])
                    Cost+=(((l * cb) + (((l*(M-1))+j) * ci) + cr) * P) + (b**(j-i)) * (dblquad(lambda w,z: (c*((l*M+j)*T-z-w))*(mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l*M+j)*T-(z+w)) * np.exp(-mi_falha*((l*M+j)*T-(z+w))), (l*M+(i-1))*T, (l*M+i)*T, lambda z:  ((l*M+(j-1))*T)-z, lambda z: ((l*M+j)*T)-z)[0])
                    Lifetime+=((((l*(M))+j) * T) * P)
                    Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 25 - Defeito menor chega por degradação e é substituído em inspeção menor
    def Scenario25():
        Probability=0
        Cost = 0
        Lifetime = 0
        for l in range(0, K):
            for i in range(1, M):
                P = (1-b) * quad(lambda x: np.exp(-mi_menor*x) * fx(x) * Ry((l*M+i)*T-x) * np.exp(-mi_maior*(((l*M+i)*T-x))), (l*M+(i-1))*T, (l*M+i)*T)[0]
                Cost += (((l * cb) + (((l*(M-1))+i) * ci) + cr) * P)
                Lifetime += ((((l*(M))+i) * T) * P)
                Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 26 - Defeito menor chega por choque e é substituído em inspeção menor
    def Scenario26():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K):
            for i in range(1,M):
                P=(1-b) * quad(lambda z: mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1)) * Ry((l*M+i)*T-z) * np.exp(-mi_maior*(((l*M+i)*T-z))), (l*M+(i-1))*T, (l*M+i)*T)[0]
                Cost+=(((l * cb) + (((l*(M-1))+i) * ci)+ cr) * P)
                Lifetime+=((((l*(M))+i) * T) * P)
                Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 27 - Defeito menor chega por degradação e é substituído em inspeção menor FALSO NEGATIVO
    def Scenario27():
        Probability=0
        Cost = 0; Lifetime = 0
        for l in range(0, K):
            for j in range(2, M):
                for i in range(1, j):
                    P = (b**(j-i)) * (1-b) * quad(lambda x: (np.exp(-mi_menor*x) * fx(x)) * Ry((l*M+j)*T-x) * np.exp(-mi_maior*((l*M+j)*T-x)), (l*M+(i-1))*T, (l*M+i)*T)[0]
                    Cost += (((l * cb) + (((l*(M-1)) + j) * ci) + cr) * P)
                    Lifetime += ((((l*(M)) + j) * T) * P)
                    Probability+=P
        return Cost, Lifetime,Probability



    # CENÁRIO 28 - Defeito menor chega por choque e é substituído em inspeção menor FALSO NEGATIVO
    def Scenario28():
        Probability=0
        Cost = 0; Lifetime = 0
        for l in range(0, K):
            for j in range(2, M):
                for i in range(1, j):
                    P = (b**(j-i)) * (1-b) * quad(lambda z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * Ry((l*M+j)*T-z) * np.exp(-mi_maior*((l*M+j)*T-z)), (l*M+(i-1))*T, (l*M+i)*T)[0]
                    Cost += (((l*cb) + (((l*(M-1)) + j) * ci) + cr) * P)
                    Lifetime += (((l*(M)) + j) * T) * P
                    Probability+=P
        return Cost, Lifetime,Probability


    # CENÁRIO 29 - Defeito menor chega por degradação, entre inspeções menores, e é substituído em inspeção maior
    def Scenario29():
        Probability=0
        Cost = 0; Lifetime = 0
        for l in range(0, K-1):
            for i in range(1, M+1):
                P = (b**(M-i)) * quad(lambda x: (np.exp(-mi_menor*x) * fx(x)) * Ry((l+1)*M*T-x) * np.exp(-mi_maior*((l+1)*M*T-x)), (l*M+(i-1))*T, (l*M+i)*T)[0]
                Cost += ((((l+1)*cb) + ((l+1)*(M-1)*ci) + cr) * P)
                Lifetime += ((l+1)*M*T) * P
                Probability+=P
        return Cost, Lifetime,Probability


    # CENÁRIO 30 - Defeito menor chega por choque e é substituído em inspeção maior
    def Scenario30():
        Probability=0
        Cost = 0; Lifetime = 0
        for l in range(0, K-1):
            for i in range(1, M+1):
                P = (b**(M-i)) * quad(lambda z: (mi_menor*np.exp(-mi_menor*z)*(np.exp(-(z/n1)**b1))) * Ry((l+1)*M*T-z) * np.exp(-mi_maior*((l+1)*M*T-z)), (l*M+(i-1))*T, (l*M+i)*T)[0]
                Cost += ((((l+1)*cb) + ((l+1)*(M-1)*ci) + cr) * P)
                Lifetime += ((l+1)*M*T) * P
                Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 31 - Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO e substituição em inspeção maior.
    def Scenario31():
        Probability=0
        Cost = 0; Lifetime = 0
        for l in range(0, K-1):
            P = dblquad(lambda y, x: (np.exp(-mi_menor*x) * fx(x)) * (np.exp(-mi_maior*y) * fy(y)) * Rh((l+1)*M*T-(x+y)) * np.exp(-mi_falha*((l+1)*M*T-(x+y))), ((l+1)*M-1)*T, (l+1)*M*T, lambda x: 0, lambda x: (l+1)*M*T - x)[0]
            Cost += ((((l+1)*cb) + ((l+1)*(M-1)*ci) + cr) * P) + dblquad(lambda y, x: (c*((l+1)*M*T-x-y))*(np.exp(-mi_menor*x) * fx(x)) * (np.exp(-mi_maior*y) * fy(y)) * Rh((l+1)*M*T-(x+y)) * np.exp(-mi_falha*((l+1)*M*T-(x+y))), ((l+1)*M-1)*T, (l+1)*M*T, lambda x: 0, lambda x: (l+1)*M*T - x)[0]
            Lifetime += ((l+1)*M*T) * P
            Probability+=P
        return Cost, Lifetime,Probability

     # CENÁRIO 32 - Defeito menor DEGRADAÇÃO, defeito maior CHOQUE e substituição em inspeção maior.
    def Scenario32():
        Probability=0
        Cost = 0; Lifetime = 0
        for l in range(0, K-1):
            P = dblquad(lambda w, x: (np.exp(-mi_menor*x) * fx(x)) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l+1)*M*T-(x+w)) * np.exp(-mi_falha*((l+1)*M*T-(x+w))), ((l+1)*M-1)*T, (l+1)*M*T, lambda x: 0, lambda x: (l+1)*M*T - x)[0]
            Cost += ((((l+1)*cb) + ((l+1)*(M-1)*ci) + cr) * P) + dblquad(lambda w, x: (c*((l+1)*M*T-x-w))*(np.exp(-mi_menor*x) * fx(x)) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l+1)*M*T-(x+w)) * np.exp(-mi_falha*((l+1)*M*T-(x+w))), ((l+1)*M-1)*T, (l+1)*M*T, lambda x: 0, lambda x: (l+1)*M*T - x)[0]
            Lifetime += ((l+1)*M*T) * P
            Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 33 - Defeito menor CHOQUE, defeito maior DEGRADAÇÃO e substituição em inspeção maior.
    def Scenario33():
        Probability=0
        Cost = 0; Lifetime = 0
        for l in range(0, K-1):
            P = dblquad(lambda y, z: (mi_menor*np.exp(-mi_menor*z)*np.exp(-(z/n1)**b1)*np.exp(-mi_maior*y)*fy(y)*Rh((l+1)*M*T-(z+y))*np.exp(-mi_falha*((l+1)*M*T-(z+y)))), ((l+1)*M-1)*T, (l+1)*M*T, lambda z: 0, lambda z: (l+1)*M*T - z)[0]
            Cost += ((((l+1)*cb)+((l+1)*(M-1)*ci)+cr)*P) + dblquad(lambda y, z: (c*((l+1)*M*T-z-y)*mi_menor*np.exp(-mi_menor*z)*np.exp(-(z/n1)**b1)*np.exp(-mi_maior*y)*fy(y)*Rh((l+1)*M*T-(z+y))*np.exp(-mi_falha*((l+1)*M*T-(z+y)))), ((l+1)*M-1)*T, (l+1)*M*T, lambda z: 0, lambda z: (l+1)*M*T - z)[0]
            Lifetime += ((l+1)*M*T)*P
            Probability+=P
        return Cost, Lifetime,Probability


    # CENÁRIO 34 - Defeito menor CHOQUE, defeito maior CHOQUE e substituição em inspeção maior.
    def Scenario34():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K-1):
            P=(dblquad(lambda w,z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l+1)*M*T-(z+w)) * np.exp(-mi_falha*(((l+1)*M*T-(z+w)))), ((l+1)*M-1)*T , (l+1)*M*T, lambda z: 0, lambda z: ((l+1)*M*T-z))[0])
            Cost+=((((l+1) * cb) + ((l+1)*(M-1) * ci)+ cr) * P) + (dblquad(lambda w,z: (c*((l+1)*M*T-z-w))*(mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l+1)*M*T-(z+w)) * np.exp(-mi_falha*(((l+1)*M*T-(z+w)))), ((l+1)*M-1)*T , (l+1)*M*T, lambda z: 0, lambda z: ((l+1)*M*T-z))[0])
            Lifetime+=(((l+1)*M * T) * P)
            Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 35 - Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO e substituição em inspeção maior APÓS UM OU MAIS FALSO NEGATIVOS.
    def Scenario35():
        Probability=0
        Cost = 0
        Lifetime = 0
        for l in range(0, K-1):
            for i in range(1, M):
                P = (b**(M-i)) * dblquad(lambda y, x: np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*Rh((l+1)*M*T-(x+y))*np.exp(-mi_falha*((l+1)*M*T-(x+y))), (l*M+(i-1))*T, (l*M+i)*T, lambda x: ((l+1)*M*T - T - x), lambda x: ((l+1)*M*T - x))[0]
                Cost += ((l+1)*cb + (l+1)*(M-1)*ci + cr)*P + (b**(M-i)) * dblquad(lambda y, x: c*((l+1)*M*T - x - y)*np.exp(-mi_menor*x)*fx(x)*np.exp(-mi_maior*y)*fy(y)*Rh((l+1)*M*T-(x+y))*np.exp(-mi_falha*((l+1)*M*T-(x+y))), (l*M+(i-1))*T, (l*M+i)*T, lambda x: ((l+1)*M*T - T - x), lambda x: ((l+1)*M*T - x))[0]
                Lifetime += ((l+1)*M*T)*P
                Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 36 - Defeito menor DEGRADAÇÃO, defeito maior CHOQUE e substituição em inspeção maior APÓS UM OU MAIS FALSO NEGATIVOS.
    def Scenario36():
        Probability=0
        Cost = 0
        Lifetime = 0
        for l in range(0, K-1):
            for i in range(1, M):
                P = (b**(M-i)) * dblquad(lambda w, x: np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*np.exp(-(w/n2)**b2)*Rh((l+1)*M*T-(x+w))*np.exp(-mi_falha*((l+1)*M*T-(x+w))), (l*M+(i-1))*T, (l*M+i)*T, lambda x: ((l+1)*M*T - T - x), lambda x: ((l+1)*M*T - x))[0]
                Cost += ((l+1)*cb + (l+1)*(M-1)*ci + cr)*P + (b**(M-i)) * dblquad(lambda w, x: c*((l+1)*M*T - x - w)*np.exp(-mi_menor*x)*fx(x)*mi_maior*np.exp(-mi_maior*w)*np.exp(-(w/n2)**b2)*Rh((l+1)*M*T-(x+w))*np.exp(-mi_falha*((l+1)*M*T-(x+w))), (l*M+(i-1))*T, (l*M+i)*T, lambda x: ((l+1)*M*T - T - x), lambda x: ((l+1)*M*T - x))[0]
                Lifetime += ((l+1)*M*T)*P
                Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 37 - Defeito menor CHOQUE, defeito maior DEGRADAÇÃO e substituição em inspeção maior APÓS UM OU MAIS FALSO NEGATIVOS.
    def Scenario37():
        Probability=0
        Cost = 0
        Lifetime = 0
        for l in range(0, K-1):
            for i in range(1, M):
                P = (b**(M-i)) * dblquad(lambda y, z: (mi_menor*np.exp(-mi_menor*z) * np.exp(-(z/n1)**b1)) * (np.exp(-mi_maior*y) * fy(y)) * Rh((l+1)*M*T - (z + y)) * np.exp(-mi_falha*((l+1)*M*T - (z + y))), (l*M+(i-1))*T, (l*M+i)*T, lambda x: ((l+1)*M*T - T - x), lambda x: ((l+1)*M*T - x))[0]
                Cost += (((l+1)*cb + (l+1)*(M-1)*ci + cr) * P + (b**(M-i)) * dblquad(lambda y, z: c*((l+1)*M*T - z - y) * (mi_menor*np.exp(-mi_menor*z) * np.exp(-(z/n1)**b1)) * (np.exp(-mi_maior*y) * fy(y)) * Rh((l+1)*M*T - (z + y)) * np.exp(-mi_falha*((l+1)*M*T - (z + y))), (l*M+(i-1))*T, (l*M+i)*T, lambda x: ((l+1)*M*T - T - x), lambda x: ((l+1)*M*T - x))[0])
                Lifetime += ((l+1)*M*T) * P
                Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 38 - Defeito menor CHOQUE, defeito maior CHOQUE e substituição em inspeção maior APÓS UM OU MAIS FALSO NEGATIVOS.
    def Scenario38():
        Probability=0
        Cost=0
        Lifetime=0
        for l in range(0,K-1):
            for i in range(1,M):
                P=(b**(M-i)) * (dblquad(lambda w,z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l+1)*M*T-(z+w)) * np.exp(-mi_falha*(((l+1)*M*T-(z+w)))), (l*M+(i-1))*T , (l*M+i)*T, lambda z: (((l+1)*M*T)-T-z), lambda z: ((l+1)*M*T-z))[0])
                Cost+=((((l+1) * cb) + ((l+1)*(M-1) * ci)+ cr) * P) + (b**(M-i)) * (dblquad(lambda w,z: (c*((l+1)*M*T-z-w)) * (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh((l+1)*M*T-(z+w)) * np.exp(-mi_falha*(((l+1)*M*T-(z+w)))), (l*M+(i-1))*T , (l*M+i)*T, lambda z: (((l+1)*M*T)-T-z), lambda z: ((l+1)*M*T-z))[0])
                Lifetime+=(((l+1)*M * T) * P)
                Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 39 -Defeito menor chega por degradação de e é substituído em inspeção maior em KMT
    def Scenario39():
        Probability=0
        Cost=0
        Lifetime=0
        for i in range(1,M+1):
            P=(b**(M-i)) * (quad(lambda x: (np.exp(-mi_menor*x) * fx(x)) * Ry(K*M*T-x) * np.exp(-mi_maior*((K*M*T-x))), (((K-1)*M+(i-1)))*T, (((K-1)*M+i))*T)[0])
            Cost+=((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P)
            Lifetime+=((K*M * T) * P)
            Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 40 -Defeito menor chega por choque de e é substituído em inspeção maior em KMT
    def Scenario40():
        Probability=0
        Cost=0
        Lifetime=0
        for i in range(1,M+1):
            P=(b**(M-i)) * (quad(lambda z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * Ry(K*M*T-z) * np.exp(-mi_maior*((K*M*T-z))), (((K-1)*M+(i-1)))*T, (((K-1)*M+i))*T)[0])
            Cost+=((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P)
            Lifetime+=((K*M * T) * P)
            Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 41 - Defeito menor por DEGRADAÇÃO na ith-1 inspeção, defeito maior DEGRADAÇÃO e substituição em KMT
    def Scenario41():
        P=(dblquad(lambda y,x: (np.exp(-mi_menor*x) * fx(x)) * (np.exp(-mi_maior*y) * fy(y)) * Rh(K*M*T-(x+y)) * np.exp(-mi_falha*((K*M*T-(x+y)))), (K*M-1)*T, K*M*T, lambda x: 0, lambda x: K*M*T-x)[0])
        Cost=((((K-1)*cb) + ((K)*(M-1)*ci)+ cr) * P) + (dblquad(lambda y,x: (c*(K*M*T-x-y)) * (np.exp(-mi_menor*x) * fx(x)) * (np.exp(-mi_maior*y) * fy(y)) * Rh(K*M*T-(x+y)) * np.exp(-mi_falha*((K*M*T-(x+y)))), (K*M-1)*T, K*M*T, lambda x: 0, lambda x: K*M*T-x)[0])
        Lifetime=((K*M*T) * P)
        return Cost, Lifetime,P

    # CENÁRIO 42 - Defeito menor por DEGRADAÇÃO na ith-1 inspeção, defeito maior CHOQUE e substituição em KMT
    def Scenario42():
        P=(dblquad(lambda w,x: (np.exp(-mi_menor*x) * fx(x)) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh(K*M*T-(x+w)) * np.exp(-mi_falha*((K*M*T-(x+w)))), (K*M-1)*T, K*M*T, lambda x: 0, lambda x: K*M*T-x)[0])
        Cost=((((K-1)*cb) + ((K)*(M-1)*ci)+ cr) * P) + (dblquad(lambda w,x: (c*(K*M*T-x-w)) * (np.exp(-mi_menor*x) * fx(x)) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh(K*M*T-(x+w)) * np.exp(-mi_falha*((K*M*T-(x+w)))), (K*M-1)*T, K*M*T, lambda x: 0, lambda x: K*M*T-x)[0])
        Lifetime=((K*M*T) * P)
        return Cost,Lifetime,P
    
    # CENÁRIO 43 - Defeito menor por CHOQUE na ith-1 inspeção, defeito maior DEGRADAÇÃO e substituição em KMT
    def Scenario43():
        P=(dblquad(lambda y,z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (np.exp(-mi_maior*y) * fy(y)) * Rh(K*M*T-(z+y)) * np.exp(-mi_falha*((K*M*T-(z+y)))), (K*M-1)*T, K*M*T, lambda z: 0, lambda z: K*M*T-z)[0])
        Cost=((((K-1)*cb) + ((K)*(M-1)*ci)+ cr) * P) + (dblquad(lambda y,z: (c*(K*M*T-z-y)) * (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (np.exp(-mi_maior*y) * fy(y)) * Rh(K*M*T-(z+y)) * np.exp(-mi_falha*((K*M*T-(z+y)))), (K*M-1)*T, K*M*T, lambda z: 0, lambda z: K*M*T-z)[0])
        Lifetime=((K*M*T) * P)
        return Cost, Lifetime,P
    
    # CENÁRIO 44 - Defeito menor por CHOQUE na ith-1 inspeção, defeito maior CHOQUE e substituição em KMT
    def Scenario44():
        P=(dblquad(lambda w,z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh(K*M*T-(z+w)) * np.exp(-mi_falha*((K*M*T-(z+w)))), (K*M-1)*T, K*M*T, lambda z: 0, lambda z: K*M*T-z)[0])
        Cost=((((K-1)*cb) + ((K)*(M-1)*ci)+ cr) * P) + (dblquad(lambda w,z: (c*(K*M*T-z-w)) * (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh(K*M*T-(z+w)) * np.exp(-mi_falha*((K*M*T-(z+w)))), (K*M-1)*T, K*M*T, lambda z: 0, lambda z: K*M*T-z)[0])
        Lifetime=((K*M*T) * P)
        return Cost,Lifetime,P
    
    # CENÁRIO 45 - Defeito menor DEGRADAÇÃO, defeito maior DEGRADAÇÃO após falso negativo e substituição em KMT após um ou mais falso negativo
    def Scenario45():
        Probability=0
        Cost=0
        Lifetime=0
        for i in range(1,M):
            P=(b**(M-i)) * (dblquad(lambda y,x: (np.exp(-mi_menor*x) * fx(x)) * (np.exp(-mi_maior*y) * fy(y)) * Rh(K*M*T-(x+y)) * np.exp(-mi_falha*((K*M*T-(x+y)))), ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda x: (K*M-1)*T-x, lambda x: K*M*T-x)[0])
            Cost+=((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P) + (b**(M-i)) * (dblquad(lambda y,x: (c*(K*M*T-x-y)) * (np.exp(-mi_menor*x) * fx(x)) * (np.exp(-mi_maior*y) * fy(y)) * Rh(K*M*T-(x+y)) * np.exp(-mi_falha*((K*M*T-(x+y)))), ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda x: (K*M-1)*T-x, lambda x: K*M*T-x)[0])
            Lifetime+=((K*M * T) * P)
            Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 46 - Defeito menor DEGRADAÇÃO, defeito maior CHOQUE após falso negativo e substituição em KMT após um ou mais falso negativo
    def Scenario46():
        Probability=0
        Cost=0
        Lifetime=0
        for i in range(1,M):
            P=(b**(M-i)) * (dblquad(lambda w,x: (np.exp(-mi_menor*x) * fx(x)) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh(K*M*T-(x+w)) * np.exp(-mi_falha*((K*M*T-(x+w)))), ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda x: (K*M-1)*T-x, lambda x: K*M*T-x)[0])
            Cost+=((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P) + (b**(M-i)) * (dblquad(lambda w,x: (c*(K*M*T-x-w)) * (np.exp(-mi_menor*x) * fx(x)) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh(K*M*T-(x+w)) * np.exp(-mi_falha*((K*M*T-(x+w)))), ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda x: (K*M-1)*T-x, lambda x: K*M*T-x)[0])
            Lifetime+=((K*M * T) * P)
            Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 47 - Defeito menor CHOQUE, defeito maior DEGRADAÇÃO após falso negativo e substituição em KMT após um ou mais falso negativo
    def Scenario47():
        Probability=0
        Cost=0
        Lifetime=0
        for i in range(1,M):
            P=(b**(M-i)) * (dblquad(lambda y,z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (np.exp(-mi_maior*y) * fy(y)) * Rh(K*M*T-(z+y)) * np.exp(-mi_falha*((K*M*T-(z+y)))), ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda z: (K*M-1)*T-z, lambda z: K*M*T-z)[0])
            Cost+=((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P) + (b**(M-i)) * (dblquad(lambda y,z: (c*(K*M*T-z-y)) *  (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (np.exp(-mi_maior*y) * fy(y)) * Rh(K*M*T-(z+y)) * np.exp(-mi_falha*((K*M*T-(z+y)))), ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda z: (K*M-1)*T-z, lambda z: K*M*T-z)[0])
            Lifetime+=((K*M * T) * P)
            Probability+=P
        return Cost, Lifetime,Probability

    # CENÁRIO 48 - Defeito menor CHOQUE, defeito maior CHOQUE após falso negativo e substituição em KMT após um ou mais falso negativo
    def Scenario48():
        Probability=0
        Cost=0
        Lifetime=0
        for i in range(1,M):
            P=(b**(M-i)) * (dblquad(lambda w,z: (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh(K*M*T-(z+w)) * np.exp(-mi_falha*((K*M*T-(z+w)))), ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda z: (K*M-1)*T-z, lambda z: K*M*T-z)[0])
            Cost+=((((K-1)* cb) + (K*(M-1) * ci)+ cr) * P) + (b**(M-i)) * (dblquad(lambda w,z: (c*(K*M*T-z-w)) *  (mi_menor*np.exp(-mi_menor*z) * (np.exp(-(z/n1)**b1))) * (mi_maior*np.exp(-mi_maior*w) * (np.exp(-(w/n2)**b2))) * Rh(K*M*T-(z+w)) * np.exp(-mi_falha*((K*M*T-(z+w)))), ((K-1)*M+(i-1))*T, ((K-1)*M+i)*T, lambda z: (K*M-1)*T-z, lambda z: K*M*T-z)[0])
            Lifetime+=((K*M * T)*P)
            Probability+=P
        return Cost, Lifetime,Probability
    
    # CENÁRIO 49 - Componente sobrevive até KMT
    def Scenario49():
        P=quad(lambda x: fx(x) * np.exp(-mi_menor*(K*M*T)), K*M*T, np.inf)[0]
        Cost=((K-1)*cb + (K*(M-1)*ci)+cr)*P
        Lifetime=((K*M*T) * P)
        return Cost,Lifetime,P
    
    C1=Scenario1()
    C2=Scenario2()
    C3=Scenario3()
    C4=Scenario4()
    C5=Scenario5()
    C6=Scenario6()
    C7=Scenario7()
    C8=Scenario8()
    C9=Scenario9()
    C10=Scenario10()
    C11=Scenario11()
    C12=Scenario12()
    C13=Scenario13()
    C14=Scenario14()
    C15=Scenario15()
    C16=Scenario16()
    C17=Scenario17()
    C18=Scenario18()
    C19=Scenario19()
    C20=Scenario20()
    C21=Scenario21()
    C22=Scenario22()
    C23=Scenario23()
    C24=Scenario24()
    C25=Scenario25()
    C26=Scenario26()
    C27=Scenario27()
    C28=Scenario28()
    C29=Scenario29()
    C30=Scenario30()
    C31=Scenario31()
    C32=Scenario32()
    C33=Scenario33()
    C34=Scenario34()
    C35=Scenario35()
    C36=Scenario36()
    C37=Scenario37()
    C38=Scenario38()
    C39=Scenario39()
    C40=Scenario40()
    C41=Scenario41()
    C42=Scenario42()
    C43=Scenario43()
    C44=Scenario44()
    C45=Scenario45()
    C46=Scenario46()
    C47=Scenario47()
    C48=Scenario48()
    C49=Scenario49()
    
    ec_total=C1[0]+C2[0]+C3[0]+C4[0]+C5[0]+C6[0]+C7[0]+C8[0]+C9[0]+C10[0]+C11[0]+C12[0]+C13[0]+C14[0]+C15[0]+C16[0]+C17[0]+C18[0]+C19[0]+C20[0]+C21[0]+C22[0]+C23[0]+C24[0]+C25[0]+C26[0]+C27[0]+C28[0]+C29[0]+C30[0]+C31[0]+C32[0]+C33[0]+C34[0]+C35[0]+C36[0]+C37[0]+C38[0]+C39[0]+C40[0]+C41[0]+C42[0]+C43[0]+C44[0]+C45[0]+C46[0]+C47[0]+C48[0]+C49[0]
    el_total=C1[1]+C2[1]+C3[1]+C4[1]+C5[1]+C6[1]+C7[1]+C8[1]+C9[1]+C10[1]+C11[1]+C12[1]+C13[1]+C14[1]+C15[1]+C16[1]+C17[1]+C18[1]+C19[1]+C20[1]+C21[1]+C22[1]+C23[1]+C24[1]+C25[1]+C26[1]+C27[1]+C28[1]+C29[1]+C30[1]+C31[1]+C32[1]+C33[1]+C34[1]+C35[1]+C36[1]+C37[1]+C38[1]+C39[1]+C40[1]+C41[1]+C42[1]+C43[1]+C44[1]+C45[1]+C46[1]+C47[1]+C48[1]+C49[1]
    # p_total=C1[2]+C2[2]+C3[2]+C4[2]+C5[2]+C6[2]+C7[2]+C8[2]+C9[2]+C10[2]+C11[2]+C12[2]+C13[2]+C14[2]+C15[2]+C16[2]+C17[2]+C18[2]+C19[2]+C20[2]+C21[2]+C22[2]+C23[2]+C24[2]+C25[2]+C26[2]+C27[2]+C28[2]+C29[2]+C30[2]+C31[2]+C32[2]+C33[2]+C34[2]+C35[2]+C36[2]+C37[2]+C38[2]+C39[2]+C40[2]+C41[2]+C42[2]+C43[2]+C44[2]+C45[2]+C46[2]+C47[2]+C48[2]+C49[2]
    return ec_total/el_total#,p_total

@njit
def MinorDefect(n1, b1, mi_menor):
    defeito_menor_degradacao = np.random.weibull(b1) * n1
    defeito_menor_choque = np.random.exponential(1 / mi_menor)
    return min(defeito_menor_degradacao, defeito_menor_choque)

@njit
def MajorDefect(n2, b2, mi_maior):
    defeito_maior_degradacao = np.random.weibull(b2) * n2
    defeito_maior_choque = np.random.exponential(1 / mi_maior)
    return min(defeito_maior_degradacao, defeito_maior_choque)

@njit
def DelayTime(l_tx, mi_falha):
    falha_degradacao = np.random.exponential(1 / l_tx)
    falha_choque = np.random.exponential(1 / mi_falha)
    return min(falha_degradacao, falha_choque)

@njit(parallel=True)
def Simulation(K, M, T, Runs, n1, b1, mi_menor, n2, b2, mi_maior, l_tx, mi_falha, b, cb, ci, cr, cf, c):
    Cost=0
    Lifetime=0
    for i in prange(Runs):
        Time=0.0
        ####Generating the random variables####################################
        X=MinorDefect(n1,b1,mi_menor)
        Y=X + MajorDefect(n2,b2,mi_maior)
        H=Y + DelayTime(l_tx, mi_falha)
        ####Checking cases#####################################################
        if (X>K*M*T): #Planned renovation at KMT
            Cost+=(K-1)*cb + K*(M-1)*ci + cr
            Time=K*M*T
        else: #Minor defect happens before the planned renovation
            Cost+=cb*(int(np.floor(np.floor(X/T)/M))) + ci*((int(np.floor(X/T)) - int(np.floor(np.floor(X/T)/M))))
            Time=round((T*int(np.floor(X/T))) + T, 6)
            ##Checking the cases before KMT####################################
            Renovation=False
            while (Time<Y and Renovation==False): #Advancing until major defect
                if (Time==K*M*T):
                    Renovation=True
                    Cost+=cr
                    break
                else:
                    Major=round(Time/T, 6)%M
                    if (Major==0):
                        Renovation=True
                        Cost+=cb + cr
                        break
                    else:
                        Random=np.random.random()
                        if (Random>b):
                            Renovation=True
                            Cost+=ci + cr
                            break
                        else:
                            Cost+=ci
                    Time+=T
                    Time=round(Time, 6)
            ##If we reach major defect#########################################
            while (Time<H and Renovation==False):
                if (Time==K*M*T):
                    Renovation=True
                    Cost+=cr + c*(Time-Y)
                    break
                else:
                    Major=round(Time/T, 6)%M
                    if (Major==0):
                        Renovation=True
                        Cost+=cb + cr + c*(Time-Y)
                        break
                    else:
                        Renovation=True
                        Cost+=ci + cr + c*(Time-Y)
                        break
            ##Checking if there is a failure###################################
            if (Renovation==False):
                Cost+=cf + c*(H-Y)
                Time=H
        Lifetime+=Time
    return Cost/Lifetime
    
def main():
    #criando 3 colunas
    col1, col2, col3= st.columns(3)
    foto = Image.open('randomen.png')
    #st.sidebar.image("randomen.png", use_column_width=True)
    #inserindo na coluna 2
    col2.image(foto, use_column_width=True)
    #O código abaixo centraliza e atribui cor
    st.markdown("<h2 style='text-align: center; color: #306754;'>AIM-SHOCK (Age-based and Inspection Maintenance under SHOCKs)</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #F3F3F3; padding: 10px; text-align: center;">
          <p style="font-size: 20px; font-weight: bold;">Multi-Level Inspection and Age-Based Maintenance under Shocks: A Simulation-Analytical Optimization Approach</p>
          <p style="font-size: 15px;">By: Victor H. R. Lima, Eugênio A. S. Fischetti & Cristiano A. V. Cavalcante</p>
        </div>
        """, unsafe_allow_html=True)

    menu = ["Cost-rate", "Information", "Website"]
    
    choice = st.sidebar.selectbox("Select here", menu)
    
    if choice == menu[0]:
        st.header(menu[0])
        st.subheader("Insert the parameter values below:")
        
        global n1, b1, mi_menor, n2, b2, mi_maior, l_tx, mi_falha, b, ci, cb, cr, c, cf
        n1=st.number_input("Insert the scale parameter for the minor defect arrival (η\u2081)", min_value = 0.0, value = 10.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the minor defect arrival.")
        b1=st.number_input("Insert the shape parameter for the minor defect arrival (β\u2081)", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter specifies the shape parameter for the Weibull distribution, representing the minor defect arrival.")
        mi_menor=st.number_input("Insert rate of the exponential distribution for shocks during the good-phase (λ\u2081)", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter specifies the mean rate for the shock arrival during the good-phase.")
        n2=st.number_input("Insert the scale parameter for the major defect arrival (η\u2082)", min_value = 0.0, value = 5.0, help="This parameter specifies the scale parameter for the Weibull distribution, representing the major defect arrival.")
        b2=st.number_input("Insert the shape parameter for the major defect arrival (β\u2082)", min_value = 1.0, max_value=5.0, value = 5.0, help="This parameter specifies the shape parameter for the Weibull distribution, representing the major defect arrival.")
        mi_maior=st.number_input("Insert rate of the exponential distribution for shocks during the minor-degradation phase (λ\u2082)", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter specifies the mean rate for the shock arrival during the minor-degradation phase.")
        l_tx=st.number_input("Insert the rate of the exponential distribution for delay-time (η\u2083)", min_value = 0.0, value = 2.0, help="This parameter defines the rate of the Exponential distribution, which governs the transition from the major defective to the failed state.")
        mi_falha=st.number_input("Insert the rate of the exponential distribution for shocks during the major-degradation phase (λ\u2083)", min_value = 1.0, max_value=5.0, value = 2.5, help="This parameter defines the rate of the Exponential distribution for shock arrival during the major-degradation phase.")
        
        b=st.number_input("Insert the false-negative probability for minor inspection (b)", min_value = 0.0, max_value=1.0, value = 0.15, help="This parameter represents the probability of not indicating a minor defect during the minor inspection when, in fact, it does exist.")
        ci=st.number_input("Insert cost of minor inspection (C_{I})", min_value = 0.0, value = 0.05, help="This parameter represents the cost of conducing a minor inspection.")
        cb=st.number_input("Insert cost of major inspection (C_{B})", min_value = 0.0, value = 0.10, help="This parameter represents the cost of conducing a major inspection.")
        cr=st.number_input("Insert cost of replacement (inspections and age-based) (C_{R})", min_value = 0.0, value = 1.0, help="This parameter represents the cost associated with preventive replacements, whether performed during inspections or when the age-based threshold is reached.")
        c=st.number_input("Insert cost of major defective by time unit (C_{C})", min_value = 0.0, value = 1.0, help="This parameter represents the unitary cost associated with the time in which the component stays in major defective state for each time unit.")
        cf=st.number_input("Insert cost of failure (C_{F})", min_value = 0.0, value = 10.0, help="This parameter represents the cost of conducing a corrective replacement.")
        
        col1, col2 = st.columns(2)
        
        st.subheader("Insert the variable values below:")
        K=int(st.text_input("Insert the number of major inspections (K)", value=4))
        if (K<1):
            K=1
        M=int(st.text_input("Insert the number of minor inspections between two major inspections or the major inspection and age-based moment (M)", value=4))
        if (M<1):
            M=1
        T=st.number_input("Insert the interval between maintenance actions (T)",value=0.8000)
        
        st.subheader("Click on one of the bottons below to run the analytical or simulation-based model:")    
        botao = st.button("Get cost-rate (analytically)")
        botao2 = st.button("Get cost-rate (simulated)")
        if botao:
            st.write("---RESULT---")
            st.write("Cost-rate", Analytics_Cost_rate(K,M,T,n1, b1, mi_menor, n2, b2, mi_maior, l_tx, mi_falha, b, cb, ci, cr, cf, c))
        else:
            if botao:
                st.write("---RESULT---")
                st.write("Cost-rate", Simulation(K, M, T, Runs, n1, b1, mi_menor, n2, b2, mi_maior, l_tx, mi_falha, b, cb, ci, cr, cf, c))
         
    if choice == menu[1]:
        st.header(menu[1])
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>This app is dedicated to compute the cost-rate for a hybrid multi-level inspection and age-based maintenance policy. We assume a single system operating under four-stage degradation process. Component renovation occurs either after a failure (corrective maintenance) or during inspections, once a defect is detected or if the age-based threshold is reached (preventive maintenance). We considered false-negative probabilities during the minor inspection to detect minor defects. All inspections are perfect for detecting major defects and the major inspection perfectly detects minor defects. Shocks immediately transitionate the state of the component, but the rate of these shocks are different for the different degradation states.</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>The app computes the cost-rate for a specific solution—defined by the number of major inspections (K), the number of minor inspections (M) and the interval between maintenance actions (T).</h6>", unsafe_allow_html=True)
        st.write("<h6 style='text-align: justify; color: Blue Jay;'>For further questions or information on finding the optimal solution, please contact one of the email addresses below.</h6>", unsafe_allow_html=True)
        
        st.write('''

v.h.r.lima@random.org.br

e.a.s.fischetti@random.org.br

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
