import numpy as np
import scipy.linalg as las
from matplotlib import use, interactive
import numpy.polynomial.chebyshev as cheby
import scipy.sparse as ss
from typing import Union
interactive(True)
use('TkAgg')
import matplotlib.pyplot as plt

from utils import chebProduct, cheb2Product, product_series, product_in_chebyshev_basis, cheb3Product
from parameters import tol_tc

def trail_pad(array_to_pad: np.ndarray,
              padding_size: int,
              padding_value: float = 0.0) \
    -> np.ndarray:

    return np.pad(array_to_pad, (0,padding_size), constant_values = padding_value)


def head_pad(array_to_pad: np.ndarray,
              padding_size: int,
              padding_value: float = 0.0) \
        -> np.ndarray:

    return np.pad(array_to_pad, (padding_size, 0), constant_values = padding_value)


def chebyshev_to_function(coefficients: np.ndarray,
                          domain: np.ndarray = np.arange(0.0, 1.01, 0.01),
                          use_builtin: bool = False) \
        -> np.ndarray:

    if use_builtin:
        to_return = cheby.chebval(domain, coefficients)
    else:
        to_return = np.zeros_like(domain)
        for order, coefficient in enumerate(coefficients):
            to_return += coefficient*np.cos(order*np.arccos(domain))

    return to_return

def new_Mlam(coefficients: np.ndarray,
         basis_index: int,
         vector_parity: int = 0,
         factor_size: int = -1,
         truncation_order: int = -1) \
        -> Union[np.ndarray|ss.csr_array]:

    if basis_index < 0:
        raise ValueError('(multiplication_operator): Negative basis order. Invalid.')

    if vector_parity not in [-1, 0, 1]:
        raise ValueError('(multiplication_operator): Invalid vector parity. Only allowed is 0, 1 or -1. Got '
                         + str(vector_parity) + '.')

    if factor_size == -1:
        factor_size = len(coefficients)

    if truncation_order == -1:
        truncation_order = len(coefficients)-1

    if np.linalg.norm(coefficients) == 0.0:
        to_return = 0.0

    else:
        if basis_index == 0:
            toeplitz = las.toeplitz(trail_pad(coefficients, factor_size))
            hankel = las.hankel(trail_pad(coefficients, factor_size))
            hankel[0,:] = np.zeros_like(hankel[0,:])
            to_return = (toeplitz + hankel + coefficients[0]*np.eye(len(hankel[:,0])))/2.0

        else:
            raise ValueError('(Mlam): Basis index not supported yet.')

    return to_return

coeffs_f = np.array([1.0, 0.5, -2.7, -1.4, 4.8])
coeffs_g = np.array([0.2, -5.3, 1.3, 4.2, -1.1])
coeffs_h = np.array([-3.8, 0.9, -1.2, -0.04, 2.3])

x = np.arange(0.0, 1.01, 0.01)
f = chebyshev_to_function(coeffs_f, x)
g = chebyshev_to_function(coeffs_g, x)
h = chebyshev_to_function(coeffs_h, x)

plt.figure()
plt.plot(x, f, label=r'$f(x)$')
plt.plot(x, g, label=r'$g(x)$')
plt.plot(x, h, label=r'$h(x)$')
plt.xlabel(r'$x$')
plt.legend()
plt.grid()
plt.title('ORIGINAL FUNCTIONS')

# ----------------------------------------------------------------------------------------------------------------------
# TWO-FUNCTION MULTIPLICATION ------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

fg = f*g

productsAND = [chebProduct(coeffs_f,coeffs_g, tol = tol_tc),    # fg
               chebProduct(coeffs_g,coeffs_f, tol = tol_tc)]    # gf

productsAND2 = [cheb2Product(coeffs_f,coeffs_g, tol = tol_tc),    # fg
                cheb2Product(coeffs_g,coeffs_f, tol = tol_tc)]    # gf

productsME = [product_in_chebyshev_basis([coeffs_f,coeffs_g]),    # fg
              product_in_chebyshev_basis([coeffs_g,coeffs_f])]    # gf

# BRUTE FORCE COEFFICIENT COMPUTATION
coeffs_fg = np.zeros(2*len(coeffs_f))
temp_f = np.concatenate((coeffs_f, np.zeros(2*len(coeffs_f))))
temp_g = np.concatenate((coeffs_g, np.zeros(2*len(coeffs_g))))
for order in range(len(coeffs_fg)):
    coeffs_fg[order] = temp_f[order]*temp_g[0] / 2.0
    for term in range(len(coeffs_f)):
        coeffs_fg[order] += (temp_g[abs(term-order)] + temp_g[term+order] - temp_g[term]*(order==0))*temp_f[term]/2.0

toeplitz = las.toeplitz(trail_pad(coeffs_g, len(coeffs_f)))
hankel = las.hankel(trail_pad(coeffs_g, len(coeffs_f)))
hankel[0,:] = np.zeros(len(hankel[0,:]))
multiplicator = new_Mlam(coeffs_g,0,0)
result = multiplicator @ trail_pad(coeffs_f, len(coeffs_f))

plt.figure()
plt.plot(x, fg, label = 'Correct')
plt.plot(x, chebyshev_to_function(productsAND[0], x), label=r'Chebyshev')
# plt.plot(x, chebyshev_to_function(productsAND[1], x), label=r'$gf$ AND')
# plt.plot(x, chebyshev_to_function(productsAND2[0], x), label=r'$fg$ AND2')
# plt.plot(x, chebyshev_to_function(productsAND2[1], x), label=r'$gf$ AND2')
# plt.plot(x, chebyshev_to_function(productsME[0], x), label=r'$fg$ ME')
# plt.plot(x, chebyshev_to_function(productsME[1], x), label=r'$gf$ ME')
# plt.plot(x, chebyshev_to_function(coeffs_fg, x), label=r'$fg$ BRUTE')
# plt.plot(x, chebyshev_to_function(result, x), label=r'$fg$ NEW')
plt.xlabel(r'$x$')
plt.legend()
plt.grid()
plt.title('TWO-PRODUCTS')

plt.figure()
plt.semilogy(x, abs(chebyshev_to_function(productsME[0], x) - fg), label=r'My product function using the old $M_\lambda$')
plt.semilogy(x, abs(chebyshev_to_function(result, x) - fg), label=r'Simple product using the new $M_\lambda$')
plt.xlabel(r'$x$')
plt.legend()
plt.grid()
plt.title('Errors w.r.t. direct multiplication')

# ----------------------------------------------------------------------------------------------------------------------
# THREE-FUNCTION MULTIPLICATION ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

fgh = f*g*h
productsAND = [cheb3Product(coeffs_f,coeffs_g,coeffs_h, tol = tol_tc),    # fg
               cheb3Product(coeffs_g,coeffs_f,coeffs_h, tol = tol_tc)]    # gf
intermediate_products = [new_Mlam(coeffs_f,0) @ new_Mlam(coeffs_g,0) @ trail_pad(coeffs_h,len(coeffs_h)),
                         new_Mlam(coeffs_g,0) @ new_Mlam(coeffs_f,0) @ trail_pad(coeffs_h,len(coeffs_h))]
product1 = new_Mlam(coeffs_g,0, factor_size = len(coeffs_h)) @ trail_pad(coeffs_h,len(coeffs_h))
product1 = new_Mlam(coeffs_f,0,factor_size = len(product1)) @ trail_pad(product1,len(coeffs_f))
product2 = new_Mlam(coeffs_f,0, factor_size=len(coeffs_h)) @ trail_pad(coeffs_h,len(coeffs_h))
product2 = new_Mlam(coeffs_g,0,factor_size = len(product2)) @ trail_pad(product2,len(coeffs_f))

plt.figure()
plt.plot(x, fgh, label = 'Correct')
plt.plot(x, chebyshev_to_function(productsAND[0], x), label=r'$fgh$ AND')
plt.plot(x, chebyshev_to_function(productsAND[1], x), label=r'$gfh$ AND')
plt.xlabel(r'$x$')
plt.legend()
plt.grid()
plt.title('THREE-PRODUCTS')

plt.figure()
plt.plot(x, fgh, label = 'Correct')
plt.plot(x, chebyshev_to_function(intermediate_products[0], x), label=r'$fgh$')
plt.plot(x, chebyshev_to_function(intermediate_products[1], x), label=r'$gfh$', ls='--')
plt.xlabel(r'$x$')
plt.legend()
plt.grid()
plt.title('Intermediate products')

plt.figure()
# plt.plot(x, fgh, label = 'Correct')
plt.semilogy(x, abs(chebyshev_to_function(product1, x) - fgh), label=r'$fgh$')
plt.semilogy(x, abs(chebyshev_to_function(product2, x) - fgh), label=r'$gfh$')
plt.xlabel(r'$x$')
plt.legend()
plt.grid()
plt.title('Fully correct products')

x = [0, 0.05, 0.5, 0.55]
plt.figure()
plt.errorbar(x[0], 0.589, 0.075, fmt='o', label='Iess et al. (2012)')
plt.errorbar(x[1], 0.375, 0.06, fmt='o', label='Goossens et al. (2024)')
plt.errorbar(x[2], 0.0450e1, 0.0080e1, fmt='o', label='Williams et al. (2014)')
plt.errorbar(x[3], 0.0387e1, 0.0025e1, fmt='o', label='Thor et al. (2020)')
# plt.legend()
plt.xticks(x, ['Iess (2012)', 'Goossens (2024)','Williams (2014)','Thor (2020)'], rotation = 15)
plt.ylabel(r"Titan's Love number, $k_2$")