import numpy as np
import math

"""
Script to compute the differential rotation coefficients for 1-curl and 2-curl equations.

For each function : 
    Δl : {-3, -2, -1, 0, 1, 2, 3} -> N = 7
    c[0] -> Δl = -3
    c[1] -> Δl = -2
    c[2] -> Δl = -1
    c[3] -> Δl = 0
    c[4] -> Δl = +1
    c[5] -> Δl = +2
    c[6] -> Δl = +3
"""

# ------------------------------------------------------------------------------------------
# 1-curl equation (r.(∇ x __))
# ------------------------------------------------------------------------------------------

# Radial profile f(r) ----------------------------------------------------------------------

def f_1C_D0P(l, m, w, offdiag) : 
    """
    1-Curl equation : 
    Coefficients for the [P_(l+Δl,m)/r] terms multiplied by the radial profile [f(r)].
    """
   
    
    if offdiag == -3 : 
        if l>=4 and l>=3+m and l+m>=3: 
            c = 3*(l + 1)*(2*l + 1)*(l**2 - 7*l + 12)*np.sqrt(((l - m - 2)*(l - m - 1)*(l + m - 2)*(l + m - 1)*(l**2 - m**2))/((16*l**4 - 64*l**3 + 56*l**2 + 16*l - 15)**2))
    elif offdiag == -1 : 
        if l>=4 and l>=1+m and l+m>=1: 
            c = (1/5)*(1 + l)*(1 + 2*l)*np.sqrt(l/(-1 + 4*l**2))*((1 + l - 2*l**2)*np.sqrt(-((l**2 - m**2)/(l - 4*l**3))) + (15*m**2*np.sqrt(-((l**2 - m**2)/(l - 4*l**3))))/(1 + l) + 3*(27 - 6*l + 3*l**2 + l**3)*(-1 + l**2 - 5*m**2)*np.sqrt((l**2 - m**2)/(l*(-1 + 4*l**2)*(9 + 9*l - 4*l**2 - 4*l**3)**2)))
        elif l == 3 : 
            c = -(4/5) * m**2 * np.sqrt(9 - m**2)
        elif l == 2 : 
            c = -2*np.sqrt(4 - m**2)*(-1 + 2*m**2)
    elif offdiag == 1 : 
        c = np.sqrt(l**2 + 2*l - m**2 + 1)*(l**5+ 6*l**4+ l**3*(3*m**2 + 13)+ 6*l**2*(2*m**2 + 5)+ l*(40 - 3*m**2)- 120*m**2)/((2*l + 3) * (4*l**2 + 8*l - 5)) 
    elif offdiag == 3 : 
        c = -3*(20*np.sqrt(l*(l**2 + 5*l + 6)) + np.sqrt(l**5*(l**2 + 5*l + 6)) + 9*np.sqrt(l**3*(l**2 + 5*l + 6)))*np.sqrt((l*(l - m + 1)*(l - m + 2)*(l - m + 3)*(l + m + 1)*(l + m + 2)*(l + m + 3))/((l + 2)*(l + 3)))/((2*l + 3)*(2*l + 5)*(2*l + 7))
    else : 
        c = 0

    return w*c

def f_1C_lho1_D0P(l, m, w, offdiag) : 
    """
    1-Curl equation : 
    Coefficients for the [lho' x P_(l+Δl,m)] terms multiplied by the radial profile [f(r)].
    """
    
    if offdiag == -3 : 
        if l>=4 and l>=3+m and l+m>=3: 
                c = -6*(2*l+1)*np.sqrt(l*(l**3 - 4*l**2 + l + 6))*np.sqrt((l**3 - 3*l**2 - l + 3)/(16*l**4 - 64*l**3 + 56*l**2 + 16*l - 15))*np.sqrt(((l-m-2)*(l-m-1)*(l+m-2)*(l+m-1)*(l**2 - m**2))/((l-2)*l*(2*l-5)*(2*l-3)*(4*l**3 - 4*l**2 - l + 1)))
    elif offdiag == -1 : 
        if l>=2 and l>=1+m and l+m>=1: 
            c = (1/10)*l*(2*l+1)*np.sqrt(l**2-1)*(-30*m**2*np.sqrt((l**2-m**2)/(l**2-1))/(l-4*l**3)+2*np.sqrt((l**2-1)*(l**2-m**2))/(l-4*l**3)+15*l*np.sqrt(1/(16*l**4-40*l**2+9))*(l**2-5*m**2-1)*np.sqrt((l**2-m**2)/(16*l**6-56*l**4+49*l**2-9))-27*(l**2-6)*np.sqrt(1/(16*l**5-40*l**3+9*l))*(l**2-5*m**2-1)*np.sqrt(-(l**2-m**2)/(-16*l**7+56*l**5-49*l**3+9*l)))
    elif offdiag == 1 : 
        c = (1/20)*(l+1)*np.sqrt(l*(l+2))*(2*l+1)*(30*(l+1)**1.5*(l*(l+2)-5*m**2)/np.sqrt(((1-2*l)**2*l*(l+1)*(l+2)*(2*l+1)**2*(2*l+3)**2*(2*l+5)**2)/((l-m+1)*(l+m+1)))+60*m**2/np.sqrt((l*(l+1)**2*(l+2)*(2*l+1)**2*(2*l+3)**2)/((l-m+1)*(l+m+1)))-54*(l*(l+2)-5)*(l*(l+2)-5*m**2)/np.sqrt(((1-2*l)**2*l*(l+1)**2*(l+2)*(2*l+1)**2*(2*l+3)**2*(2*l+5)**2)/((l-m+1)*(l+m+1)))-4*np.sqrt(l*(l+2)*(l-m+1)*(l+m+1))/((l+1)*(2*l+1)*(2*l+3)))
    elif offdiag == 3 : 
        c = -(6 * l * (4 + l) * np.sqrt((1 + l - m) * (2 + l - m) * (3 + l - m) * (1 + l + m) * (2 + l + m) * (3 + l + m))) / ((3 + 2*l) * (5 + 2*l) * (7 + 2*l))
    else : 
        c = 0

    return w*c

def f_1C_D1P(l, m, w, offdiag) : 
    """
    1-Curl equation : 
    Coefficients for the [P'_(l+Δl,m)] terms multiplied by the radial profile [f(r)].
    """
    
    if offdiag == -3 : 
        if l>=4 and l>=3+m and l+m>=3: 
            c = -6*(2*l+1)*np.sqrt(l*(l**3 - 4*l**2 + l + 6))*np.sqrt((l**3 - 3*l**2 - l + 3)/(16*l**4 - 64*l**3 + 56*l**2 + 16*l - 15))*np.sqrt(((l-m-2)*(l-m-1)*(l+m-2)*(l+m-1)*(l**2 - m**2))/((l-2)*l*(2*l-5)*(2*l-3)*(4*l**3 - 4*l**2 - l + 1)))
    elif offdiag == -1 : 
        if l>=2 and l>=1+m and l+m>=1: 
            c = (1/10)*l*(2*l+1)*np.sqrt(l**2-1)*(-30*m**2*np.sqrt((l**2-m**2)/(l**2-1))/(l-4*l**3)+2*np.sqrt((l**2-1)*(l**2-m**2))/(l-4*l**3)+15*l*np.sqrt(1/(16*l**4-40*l**2+9))*(l**2-5*m**2-1)*np.sqrt((l**2-m**2)/(16*l**6-56*l**4+49*l**2-9))-27*(l**2-6)*np.sqrt(1/(16*l**5-40*l**3+9*l))*(l**2-5*m**2-1)*np.sqrt(-(l**2-m**2)/(-16*l**7+56*l**5-49*l**3+9*l)))
    elif offdiag == 1 : 
        c = (1/20)*(l+1)*np.sqrt(l*(l+2))*(2*l+1)*(30*(l+1)**1.5*(l*(l+2)-5*m**2)/np.sqrt(((1-2*l)**2*l*(l+1)*(l+2)*(2*l+1)**2*(2*l+3)**2*(2*l+5)**2)/((l-m+1)*(l+m+1)))+60*m**2/np.sqrt((l*(l+1)**2*(l+2)*(2*l+1)**2*(2*l+3)**2)/((l-m+1)*(l+m+1)))-54*(l*(l+2)-5)*(l*(l+2)-5*m**2)/np.sqrt(((1-2*l)**2*l*(l+1)**2*(l+2)*(2*l+1)**2*(2*l+3)**2*(2*l+5)**2)/((l-m+1)*(l+m+1)))-4*np.sqrt(l*(l+2)*(l-m+1)*(l+m+1))/((l+1)*(2*l+1)*(2*l+3)))
    elif offdiag == 3 : 
        c = -(6 * l * (4 + l) * np.sqrt((1 + l - m) * (2 + l - m) * (3 + l - m) * (1 + l + m) * (2 + l + m) * (3 + l + m))) / ((3 + 2*l) * (5 + 2*l) * (7 + 2*l))
    else : 
        c = 0

    return w*c


def f_1C_D0T(l, m, w, offdiag) : 
    """
    1-Curl equation : 
    Coefficients for the [T_(l+Δl,m)] terms multiplied by the radial profile [f(r)].
    """
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = 0.5*(-3)*1j*(2*l+1)*np.sqrt(l*(l+1)*(l**2-3*l+2))*m*(2*(l+4)*np.sqrt(1/(8*l**3-12*l**2-2*l+3))*np.sqrt(((l**2-m**2)*(l**2-2*l-m**2+1))/(l*(2*l**2-7*l+6)*(4*l**4-5*l**2+1)))-np.sqrt((l**2-l-2)/(8*l**3-12*l**2-2*l+3))*np.sqrt((l**4-2*l**3-2*l**2*m**2+l**2+2*l*m**2+m**4-m**2)/(l*(8*l**4-20*l**3+10*l**2+5*l-3))))
    elif offdiag == 0 : 
        if l==1 : 
            c = 1j*m*(5*l**4 + 10*l**3 - l**2*(15*m**2 + 14) - l*(15*m**2 + 19) + 45*m**2 + 3)/(5*(4*l**2 + 4*l - 3))
        else : 
            c = (1j*m*(6 + 2*l**3 + l**4 + 36*m**2 - l**2*(19 + 3*m**2) - l*(20 + 3*m**2)))/(-3 + 4*l + 4*l**2)
    elif offdiag == 2 : 
        c = (3*1j*(-6 + 5*l + l**2)*m*np.sqrt(4 + 6*l**3 + l**4 - 5*m**2 + m**4 + l**2*(13 - 2*m**2) - 6*l*(-2 + m**2)))/(30 + 32*l + 8*l**2)
    else : 
        c = 0  

    return w*c

# Radial profile r f'(r) -------------------------------------------------------------------

def df_1C_D0P(l, m, w, offdiag) : 
    """
    1-Curl equation : 
    Coefficients for the [P_(l+Δl,m)/r] terms multiplied by the radial profile [r x f'(r)].
    """
    
    if offdiag == -3 : 
        if l>=4 and l>=3+m and l+m>=3: 
            c = (3/2)*(l-3)*(l-2)*(l+1)*(2*l+1)*np.sqrt(l*(l**2-3*l+2)/(16*l**4-64*l**3+56*l**2+16*l-15))*np.sqrt(((l-m-2)*(l-m-1)*(l+m-2)*(l+m-1)*(l**2-m**2))/((l-2)*l*(2*l-5)*(2*l-3)*(4*l**3-4*l**2-l+1)))
    elif offdiag == -1 : 
        if l>=2 and l>=1+m and l+m>=1: 
            c = -(1/10)*l*np.sqrt(l*(l+1))*(2*l**2 - l - 1)*(2*np.sqrt((l+1)*(l**2 - m**2)/(l*(1 - 4*l**2)**2)) - 3*(l+6)*np.sqrt((l-1)/(16*l**4 - 40*l**2 + 9))*(l**2 - 5*m**2 - 1)*np.sqrt(-(l**2 - m**2)/(-16*l**7 + 56*l**5 - 49*l**3 + 9*l)))
    elif offdiag == 1 : 
        c = ((2 + 3*l + l**2)*np.sqrt(1 + 2*l + l**2 - m**2)*(5*l**2 + l**3 - 15*m**2 + l*(4 + 3*m**2)))/(2*(3 + 2*l)*(-5 + 8*l + 4*l**2))
    elif offdiag == 3 : 
        c = -(3*l*(3+l)*(4+l)*np.sqrt((1+l-m)*(2+l-m)*(3+l-m)*(1+l+m)*(2+l+m)*(3+l+m)))/(2*(3+2*l)*(5+2*l)*(7+2*l))
    else : 
        c = 0

    return w*c


# ------------------------------------------------------------------------------------------
# 2-curl equation (r.(∇ x (∇ x __)))
# ------------------------------------------------------------------------------------------

# Radial profile f(r) ----------------------------------------------------------------------

def f_2C_D0P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [P_(l+Δl,m)/r^2] terms multiplied by the radial profile [f(r)].
    """
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = (3/2)*1j*l*(l+1)*(2*l+1)*m*((l-2)*(l-1)**2*np.sqrt(1/(8*l**4-20*l**3+10*l**2+5*l-3))*np.sqrt((l**4-2*l**3-2*l**2*m**2+l**2+2*l*m**2+m**4-m**2)/(8*l**4-20*l**3+10*l**2+5*l-3)) - 2*(l+1)*np.sqrt(l**2-3*l+2)*np.sqrt(l/(8*l**4-4*l**3-14*l**2+l+3))*np.sqrt(((l**2-m**2)*(l**2-2*l-m**2+1))/(l*(2*l**2-7*l+6)*(4*l**4-5*l**2+1))))
    elif offdiag == 0 : 
        if l==1 : 
            c = (1j*l*(l+1)*m*(5*l**4 + 10*l**3 + l**2*(13 - 15*m**2) + l*(8 - 15*m**2) - 6))/(5*(4*l**2 + 4*l - 3))
        else : 
            c = -(1j*np.sqrt(l*(l+1))*m*(-8*np.sqrt(l**19*(l+1)) - 36*np.sqrt(l**17*(l+1)) - 26*np.sqrt(l**15*(l+1)) + 77*np.sqrt(l**13*(l+1)) + 91*np.sqrt(l**11*(l+1)) - 49*np.sqrt(l**9*(l+1)) - 69*np.sqrt(l**7*(l+1)) + 8*np.sqrt(l**5*(l+1)) - 3*(-8*np.sqrt(l**15*(l+1)) - 28*np.sqrt(l**13*(l+1)) + 2*np.sqrt(l**11*(l+1)) + 75*np.sqrt(l**9*(l+1)) + 16*np.sqrt(l**7*(l+1)) - 65*np.sqrt(l**5*(l+1)) + 12*np.sqrt(l*(l+1)))*m**2 + 12*l*np.sqrt(l*(l+1))*(m**2 + 1)))/((4*l**2 + 4*l - 3)**2*(2*l**3 + 3*l**2 - 3*l - 2))
    elif offdiag == 2 : 
        c = (3*1j*l*(l+1)**2*(l+4)*m*np.sqrt((l-m+1)*(l-m+2)*(l+m+1)*(l+m+2)))/(2*(2*l+3)*(2*l+5))
    else : 
        c = 0

    return w*c


def f_2C_D1P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [P'_(l+Δl,m)/r] terms multiplied by the radial profile [f(r)].
    """
    
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = -(1/5)*1j*np.sqrt(6)*l*(l+1)*(2*l+1)*np.sqrt((l**2-3*l+2)/(8*l**5-4*l**4-14*l**3+l**2+3*l))*(-1)**(m-l)*(5*np.sqrt(3/2)*m*(-1)**(m-l)*np.sqrt(((l**2-l-2)*(l-m-1)*(l-m)*(l+m-1)*(l+m))/((l-1)*l*(2*l-3)*(2*l-1)*(2*l+1))) + 15*np.sqrt(6)*(l-2)*m*(-1)**(m-l)*np.sqrt(((l-m-1)*(l-m)*(l+m-1)*(l+m))/((l-2)*(l-1)*l*(l+1)*(2*l-3)*(2*l-1)*(2*l+1))))
    elif offdiag == 0 : 
        if l==1 : 
            c = (2*1j*m*(-5*l**4 - 10*l**3 + l**2*(15*m**2 + 14) + l*(15*m**2 + 19) - 45*m**2 - 3))/(5*(4*l**2 + 4*l - 3))
        else : 
            c = -(2*1j*m*(l**4 + 2*l**3 - l**2*(3*m**2 + 19) - l*(3*m**2 + 20) + 36*m**2 + 6))/(4*l**2 + 4*l - 3)
    elif offdiag == 2 : 
        c = -(3*1j*(-18 - 3*l + l**2)*m*np.sqrt((1+l-m)*(2+l-m)*(1+l+m)*(2+l+m)))/((3+2*l)*(5+2*l))
    else : 
        c = 0

    return w*c


def f_2C_D0T(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [T_(l+Δl,m)/r] terms multiplied by the radial profile [f(r)].
    """

    if offdiag == -3 : 
        if l>=4 and l>=3+m and l+m>=3: 
            c = 3*(2*l+1)*(l**2-1)*np.sqrt(l**2-5*l+6)*np.sqrt(l*(l**2-4*l+3)/(16*l**4-64*l**3+56*l**2+16*l-15))*np.sqrt(((l-m-2)*(l-m-1)*(l+m-2)*(l+m-1)*(l**2-m**2))/((l-2)*l*(2*l-5)*(2*l-3)*(4*l**3-4*l**2-l+1)))
    elif offdiag == -1 : 
        if l>=2 and l>=1+m and l+m>=1: 
            c = -((l-1)*(l+1)*(2*l+1)*np.sqrt(l/(4*l**3 + 4*l**2 - 9*l - 9))*(-15*m**2*np.sqrt((4*l**2-9)*(l**2-m**2)/((l-1)**2*l*(l+1))) + (2*l+1)*np.sqrt((4*l**3 + 4*l**2 - 9*l - 9)*(l**2-m**2)/l) - 3*(l**3-12*l**2-6*l+27)*(l**2-5*m**2-1)*np.sqrt((l**2-m**2)/((l-1)**2*l*(4*l**3 + 4*l**2 - 9*l - 9)))))/(5*(4*l**2-1))
    elif offdiag == 1 : 
        c = (np.sqrt(l**2 + 2*l - m**2 + 1)*(l**5 - 3*l**4 + l**3*(3*m**2 - 23) + 3*l**2*(19*m**2 - 5) + l*(87*m**2 + 22) - 75*m**2))/((2*l + 3)*(4*l**2 + 8*l - 5))
    elif offdiag == 3 : 
        c = -(3*l*(2+l)*(4+l)*np.sqrt((1+l-m)*(2+l-m)*(3+l-m)*(1+l+m)*(2+l+m)*(3+l+m)))/((3+2*l)*(5+2*l)*(7+2*l))
    else : 
        c = 0

    return w*c

def f_2C_lho1_D0P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [lho' x P_(l+Δl,m)/r] terms multiplied by the radial profile [f(r)].
    """
    
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = 1.5j*(2*l+1)*np.sqrt(l**2-3*l+2)*m*((-(l+1)*np.sqrt((l-2)/(8*l**3-12*l**2-2*l+3))*np.sqrt((l**4-2*l**3-2*l**2*m**2+l**2+2*l*m**2+m**4-m**2)/(8*l**4-20*l**3+10*l**2+5*l-3)) - 2*(-5*np.sqrt(l*(l+1)/(8*l**3-12*l**2-2*l+3)) + np.sqrt(l**3*(l+1)/(8*l**3-12*l**2-2*l+3)) + np.sqrt(1/(8*l**4-4*l**3-14*l**2+l+3))*l**1.5 + 2*np.sqrt(1/(8*l**4-4*l**3-14*l**2+l+3))*l**2.5 + np.sqrt(1/(8*l**4-4*l**3-14*l**2+l+3))*l**3.5) * np.sqrt(((l**2-m**2)*(l**2-2*l-m**2+1))/(l*(2*l**2-7*l+6)*(4*l**4-5*l**2+1)))))
    elif offdiag == 0 : 
        if l==1 : 
            c = 1j*m*(3*l**4 + 6*l**3 + l**2*(15*m**2 + 16) + l*(15*m**2 + 13) - 45*m**2 - 3)/(5*(4*l**2 + 4*l - 3))
        else : 
            c = -1j*m*(3*l**4 + 6*l**3 - l**2*(9*m**2 + 17) - l*(9*m**2 + 20) + 36*m**2 + 6)/(4*l**2 + 4*l - 3)
    elif offdiag == 2 : 
        c = -9j*(l**2 + l - 4)*m*np.sqrt(l**4 + 6*l**3 + l**2*(13 - 2*m**2) - 6*l*(m**2 - 2) + m**4 - 5*m**2 + 4)/(8*l**2 + 32*l + 30)
    else : 
        c = 0
    
    return w*c

def f_2C_lho2_D0P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [lho'' x P_(l+Δl,m)] terms multiplied by the radial profile [f(r)].
    """
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = (3/2)*1j*(2*l+1)*np.sqrt(l*(l+1)*(l**2-3*l+2))*m*(-2*(l-5)*np.sqrt(1/(8*l**3-12*l**2-2*l+3))*np.sqrt(((l**2-m**2)*(l**2-2*l-m**2+1))/(l*(2*l**2-7*l+6)*(4*l**4-5*l**2+1))) - np.sqrt((l**2-l-2)/(8*l**3-12*l**2-2*l+3))*np.sqrt((l**4-2*l**3-2*l**2*m**2+l**2+2*l*m**2+m**4-m**2)/(l*(8*l**4-20*l**3+10*l**2+5*l-3))))
    elif offdiag == 0 : 
        if l==1 : 
            c = (1j*m*(-5*l**4 - 10*l**3 + l**2*(15*m**2 + 14) + l*(15*m**2 + 19) - 45*m**2 - 3))/(5*(4*l**2 + 4*l - 3))
        else : 
            c = -(1j*m*(l**4 + 2*l**3 - l**2*(3*m**2 + 19) - l*(3*m**2 + 20) + 36*m**2 + 6))/(4*l**2 + 4*l - 3)
    elif offdiag == 2 : 
        c = -(3*1j*(l**2 + l - 12)*m*np.sqrt(l**4 + 6*l**3 + l**2*(13 - 2*m**2) - 6*l*(m**2 - 2) + m**4 - 5*m**2 + 4))/(8*l**2 + 32*l + 30)
    else : 
        c = 0

    return w*c


def f_2C_lho1_D1P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [lho' x P'_(l+Δl,m)] terms multiplied by the radial profile [f(r)].
    """
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = (3/2)*1j*(2*l+1)*np.sqrt(l*(l+1)*(l**2-3*l+2))*m*(-2*(l-5)*np.sqrt(1/(8*l**3-12*l**2-2*l+3))*np.sqrt(((l**2-m**2)*(l**2-2*l-m**2+1))/(l*(2*l**2-7*l+6)*(4*l**4-5*l**2+1))) - np.sqrt((l**2-l-2)/(8*l**3-12*l**2-2*l+3))*np.sqrt((l**4-2*l**3-2*l**2*m**2+l**2+2*l*m**2+m**4-m**2)/(l*(8*l**4-20*l**3+10*l**2+5*l-3))))
    elif offdiag == 0 : 
        if l==1 : 
            c = (1j*m*(-5*l**4 - 10*l**3 + l**2*(15*m**2 + 14) + l*(15*m**2 + 19) - 45*m**2 - 3))/(5*(4*l**2 + 4*l - 3))
        else : 
            c = -(1j*m*(l**4 + 2*l**3 - l**2*(3*m**2 + 19) - l*(3*m**2 + 20) + 36*m**2 + 6))/(4*l**2 + 4*l - 3)
    elif offdiag == 2 : 
        c = -(3*1j*(l**2 + l - 12)*m*np.sqrt(l**4 + 6*l**3 + l**2*(13 - 2*m**2) - 6*l*(m**2 - 2) + m**4 - 5*m**2 + 4))/(8*l**2 + 32*l + 30)
    else : 
        c = 0

    return w*c



def f_2C_D2P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [P''_(l+Δl,m)] terms multiplied by the radial profile [f(r)].
    """
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = (3/2)*1j*(2*l+1)*np.sqrt(l*(l+1)*(l**2-3*l+2))*m*(-2*(l-5)*np.sqrt(1/(8*l**3-12*l**2-2*l+3))*np.sqrt(((l**2-m**2)*(l**2-2*l-m**2+1))/(l*(2*l**2-7*l+6)*(4*l**4-5*l**2+1))) - np.sqrt((l**2-l-2)/(8*l**3-12*l**2-2*l+3))*np.sqrt((l**4-2*l**3-2*l**2*m**2+l**2+2*l*m**2+m**4-m**2)/(l*(8*l**4-20*l**3+10*l**2+5*l-3))))
    elif offdiag == 0 : 
        if l==1 : 
            c = (1j*m*(-5*l**4 - 10*l**3 + l**2*(15*m**2 + 14) + l*(15*m**2 + 19) - 45*m**2 - 3))/(5*(4*l**2 + 4*l - 3))
        else : 
            c = -(1j*m*(l**4 + 2*l**3 - l**2*(3*m**2 + 19) - l*(3*m**2 + 20) + 36*m**2 + 6))/(4*l**2 + 4*l - 3)
    elif offdiag == 2 : 
        c = -(3*1j*(l**2 + l - 12)*m*np.sqrt(l**4 + 6*l**3 + l**2*(13 - 2*m**2) - 6*l*(m**2 - 2) + m**4 - 5*m**2 + 4))/(8*l**2 + 32*l + 30)
    else : 
        c = 0

    return w*c


def f_2C_D1T(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [T'_(l+Δl,m)] terms multiplied by the radial profile [f(r)].
    """
    
    if offdiag == -3 : 
        if l>=4 and l>=3+m and l+m>=3: 
            c = -3*(2*l+1)*np.sqrt(l*(l**3-4*l**2+l+6))*np.sqrt((l**3-3*l**2-l+3)/(16*l**4-64*l**3+56*l**2+16*l-15))*np.sqrt(((l-m-2)*(l-m-1)*(l+m-2)*(l+m-1)*(l**2-m**2))/((l-2)*l*(2*l-5)*(2*l-3)*(4*l**3-4*l**2-l+1)))
    elif offdiag == -1 : 
        if l>=2 and l>=1+m and l+m>=1: 
            c = (1/10)*l*(2*l+1)*np.sqrt(l**2-1)*((-30*m**2*np.sqrt((l**2-m**2)/(l**2-1)))/(l-4*l**3) + (2*np.sqrt((l**2-1)*(l**2-m**2)))/(l-4*l**3) - 15*l*np.sqrt(1/(16*l**4-40*l**2+9))*(l**2-5*m**2-1)*np.sqrt((l**2-m**2)/(16*l**6-56*l**4+49*l**2-9)) - 27*(l**2-6)*np.sqrt(1/(16*l**5-40*l**3+9*l))*(l**2-5*m**2-1)*np.sqrt(-(l**2-m**2)/(-16*l**7+56*l**5-49*l**3+9*l)))
    elif offdiag == 1 : 
        c = (1/20)*(l+1)*np.sqrt(l*(l+2))*(2*l+1)*((-4*np.sqrt(l*(l+2)*(l**2+2*l-m**2+1)))/(4*l**3+12*l**2+11*l+3) + 60*m**2/np.sqrt(l*(l+2)*(4*l**3+12*l**2+11*l+3)**2/(l**2+2*l-m**2+1)) - 30*(l+1)**1.5*(l**2+2*l-5*m**2)/np.sqrt(l*(l**2+3*l+2)*(16*l**4+64*l**3+56*l**2-16*l-15)**2/(l**2+2*l-m**2+1)) - 54*(l**2+2*l-5)*(l**2+2*l-5*m**2)/np.sqrt(l*(l+2)*(16*l**5+80*l**4+120*l**3+40*l**2-31*l-15)**2/(l**2+2*l-m**2+1)))
    elif offdiag == 3 : 
        c = -(3*l*(l+4)*np.sqrt((l-m+1)*(l-m+2)*(l-m+3)*(l+m+1)*(l+m+2)*(l+m+3)))/((2*l+3)*(2*l+5)*(2*l+7))
    else : 
        c = 0

    return w*c

# Radial profile r f'(r) -------------------------------------------------------------------

def df_2C_D0P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [P_(l+Δl,m)/r^2] terms multiplied by the radial profile [r f'(r)].
    """
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = (3*1j*(l**2 - 5*l + 9)*m*np.sqrt(l**4 - 2*l**3 - 2*l**2*m**2 + l**2 + 2*l*m**2 + m**4 - m**2))/(4*l**2 - 8*l + 3)
    elif offdiag == 0 : 
        if l==1 : 
            c = (1j*m*(-3 - 34*l**3 - 17*l**4 - 45*m**2 + l**2*(11 + 15*m**2) + l*(28 + 15*m**2)))/(5*(-3 + 4*l + 4*l**2))
        else : 
            c = (2*1j*m*(l**4 + 2*l**3 + l**2*(11 - 3*m**2) + l*(10 - 3*m**2) - 3*(6*m**2 + 1)))/(4*l**2 + 4*l - 3)
    elif offdiag == 2 : 
        c = (3*1j*(l**2 + 7*l + 15)*m*np.sqrt(l**4 + 6*l**3 + l**2*(13 - 2*m**2) - 6*l*(m**2 - 2) + m**4 - 5*m**2 + 4))/(4*l**2 + 16*l + 15)
    else : 
        c = 0

    return w*c


def df_2C_D1P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [P'_(l+Δl,m)/r] terms multiplied by the radial profile [r f'(r)].
    """
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = -(3*1j*(2*l - 7)*m*np.sqrt(l**4 - 2*l**3 - 2*l**2*m**2 + l**2 + 2*l*m**2 + m**4 - m**2))/(4*l**2 - 8*l + 3)
    elif offdiag == 0 : 
        if l==1 : 
            c = (1j*m*(-9*l**4 - 18*l**3 + l**2*(15*m**2 + 13) + l*(15*m**2 + 22) - 45*m**2 - 3))/(5*(4*l**2 + 4*l - 3))
        else : 
            c = (2*1j*m*(10*l**2 + 10*l - 18*m**2 - 3))/(4*l**2 + 4*l - 3)
    elif offdiag == 2 : 
        c = (3*1j*(2*l + 9)*m*np.sqrt(l**4 + 6*l**3 + l**2*(13 - 2*m**2) - 6*l*(m**2 - 2) + m**4 - 5*m**2 + 4))/(4*l**2 + 16*l + 15)
    else : 
        c = 0
     
    return w*c


def df_2C_D0T(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [T_(l+Δl,m)/r] terms multiplied by the radial profile [r f'(r)].
    """
    if offdiag == -3 : 
        if l>=4 and l>=3+m and l+m>=3: 
            c = -3*(2*l+1)*np.sqrt(l*(l**3-4*l**2+l+6))*np.sqrt((l**3-3*l**2-l+3)/(16*l**4-64*l**3+56*l**2+16*l-15))*np.sqrt(((l-m-2)*(l-m-1)*(l+m-2)*(l+m-1)*(l**2-m**2))/((l-2)*l*(2*l-5)*(2*l-3)*(4*l**3-4*l**2-l+1)))
    elif offdiag == -1 : 
        if l>=2 and l>=1+m and l+m>=1: 
            c = (1/10)*l*(2*l+1)*np.sqrt(l**2-1)*((-30*m**2*np.sqrt((l**2-m**2)/(l**2-1)))/(l-4*l**3) + (2*np.sqrt((l**2-1)*(l**2-m**2)))/(l-4*l**3) - 15*l*np.sqrt(1/(16*l**4-40*l**2+9))*(l**2-5*m**2-1)*np.sqrt((l**2-m**2)/(16*l**6-56*l**4+49*l**2-9)) - 27*(l**2-6)*np.sqrt(1/(16*l**5-40*l**3+9*l))*(l**2-5*m**2-1)*np.sqrt(-(l**2-m**2)/(-16*l**7+56*l**5-49*l**3+9*l)))
    elif offdiag == 1 : 
        c = (1/20)*(l+1)*np.sqrt(l*(l+2))*(2*l+1)*((-4*np.sqrt(l*(l+2)*(l**2+2*l-m**2+1)))/(4*l**3+12*l**2+11*l+3) + 60*m**2/np.sqrt(l*(l+2)*(4*l**3+12*l**2+11*l+3)**2/(l**2+2*l-m**2+1)) - 30*(l+1)**1.5*(l**2+2*l-5*m**2)/np.sqrt(l*(l**2+3*l+2)*(16*l**4+64*l**3+56*l**2-16*l-15)**2/(l**2+2*l-m**2+1)) - 54*(l**2+2*l-5)*(l**2+2*l-5*m**2)/np.sqrt(l*(l+2)*(16*l**5+80*l**4+120*l**3+40*l**2-31*l-15)**2/(l**2+2*l-m**2+1)))
    elif offdiag == 3 : 
        c = -(3*l*(l+4)*np.sqrt((l-m+1)*(l-m+2)*(l-m+3)*(l+m+1)*(l+m+2)*(l+m+3)))/((2*l+3)*(2*l+5)*(2*l+7))
    else : 
        c = 0

    return w*c


def df_2C_lho1_D0P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [lho' x P_(l+Δl,m)/r] terms multiplied by the radial profile [r f'(r)].
    """
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = 1.5j*(2*l+1)*np.sqrt(l*(l+1)*(l**2-3*l+2))*m*(-2*(l-5)*np.sqrt(1/(8*l**3-12*l**2-2*l+3))*np.sqrt(((l**2-m**2)*(l**2-2*l-m**2+1))/(l*(2*l**2-7*l+6)*(4*l**4-5*l**2+1))) - np.sqrt((l**2-l-2)/(8*l**3-12*l**2-2*l+3))*np.sqrt((l**4-2*l**3-2*l**2*m**2+l**2+2*l*m**2+m**4-m**2)/(l*(8*l**4-20*l**3+10*l**2+5*l-3))))
    elif offdiag == 0 : 
        if l==1 : 
            c = 1j*m*(-5*l**4 - 10*l**3 + l**2*(15*m**2 + 14) + l*(15*m**2 + 19) - 45*m**2 - 3)/(5*(4*l**2 + 4*l - 3))
        else : 
            c = -1j*m*(l**4 + 2*l**3 - l**2*(3*m**2 + 19) - l*(3*m**2 + 20) + 36*m**2 + 6)/(4*l**2 + 4*l - 3)
    elif offdiag == 2 : 
        c = -3j*(l**2 + l - 12)*m*np.sqrt(l**4 + 6*l**3 + l**2*(13 - 2*m**2) - 6*l*(m**2 - 2) + m**4 - 5*m**2 + 4)/(8*l**2 + 32*l + 30)
    else : 
        c = 0

    return w*c


# Radial profile r^2 f''(r) ----------------------------------------------------------------

def d2f_2C_D0P(l, m, w, offdiag) : 
    """
    2-Curl equation : 
    Coefficients for the [P_(l+Δl,m)/r^2] terms multiplied by the radial profile [r^2 f''(r)].
    """
    
    if offdiag == -2 : 
        if l>=3 and l>=2+m and l+m>=2: 
            c = (3/2)*1j*(l-2)*(l-1)*(2*l+1)*np.sqrt(l*(l+1)*(l**2 - 3*l + 2)/(8*l**3 - 12*l**2 - 2*l + 3))*m*np.sqrt((l**2 - m**2)*(l**2 - 2*l - m**2 + 1)/(l*(2*l**2 - 7*l + 6)*(4*l**4 - 5*l**2 + 1)))
    elif offdiag == 0 : 
        if l==1 : 
            c = -(1/5)*1j*l*(l+1)*m
        else : 
            c = (1j*np.sqrt(l*(l+1))*m*(2*np.sqrt(l**11*(l+1)) + 5*np.sqrt(l**9*(l+1)) - 5*np.sqrt(l**5*(l+1)) + (-6*np.sqrt(l**7*(l+1)) - 9*np.sqrt(l**5*(l+1)) + 6*np.sqrt(l*(l+1)))*m**2 + l*np.sqrt(l*(l+1))*(9*m**2 - 2)))/(8*l**5 + 20*l**4 - 6*l**3 - 29*l**2 + l + 6)
    elif offdiag == 2 : 
        c = (3*1j*(l+2)*(l+3)*m*np.sqrt(l**4 + 6*l**3 + l**2*(13 - 2*m**2) - 6*l*(m**2 - 2) + m**4 - 5*m**2 + 4))/(8*l**2 + 32*l + 30)
    else : 
        c = 0
    return w*c
