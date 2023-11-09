from math import *
import cudf as cd
import cupy as cp
from numba import cuda
import pandas as pd
import numpy as np

# This code was adapted from Joel Ornstein's (2011 July) pure python port of 
# Aaron Cluaset's code: origninal source Source: http://www.santafe.edu/~aaronc/powerlaws/
# This adapted code uses pandas series as the primary data structure and pandas series methods
# in place of the pure python map/filter/reduce/lamda functions

# function [alpha, xmin, L]=plfit(x)
# PLFIT fits a power-law distributional model to data.
#    Source: http://www.santafe.edu/~aaronc/powerlaws/
#
#    PLFIT(x) estimates x_min and alpha according to the goodness-of-fit
#    based method described in Clauset, Shalizi, Newman (2007). x is a
#    vector of observations of some quantity to which we wish to fit the
#    power-law distribution p(x) ~  x^-alpha for x >= xmin.
#
#    The fitting procedure works as follows:
#    1) For each possible choice of x_min, we estimate alpha via the
#       method of maximum likelihood, and calculate the Kolmogorov-Smirnov
#       goodness-of-fit statistic D.
#    2) We then select as our estimate of x_min, the value that gives the
#       minimum value D over all values of x_min.
#
#    Note that this procedure gives no estimate of the uncertainty of the
#    fitted parameters, nor of the validity of the fit.
#
#    Example:
#       x = cd.Series([500,150,90,81,75,75,70,65,60,58,49,47,40])
#       [alpha, xmin, L] = plfit(x)
#   or  a = plfit(x)
#
#    The output 'alpha' is the maximum likelihood estimate of the scaling
#    exponent, 'xmin' is the estimate of the lower bound of the power-law
#    behavior, and L is the log-likelihood of the data x>=xmin under the
#    fitted power law.
#
#
# The 'zeta' helper function is modified from the open-source library 'mpmath'
#   mpmath: a Python library for arbitrary-precision floating-point arithmetic
#   http://code.google.com/p/mpmath/
#   version 0.17 (February 2011) by Fredrik Johansson and others

def plfit(x):

	pd_vec = pd.Series(np.arange(1.50,3.51,0.01))
	pd_zvec = pd_vec.apply(zeta)
	cudf_zvec = cd.from_pandas(pd_zvec)
	zvec = cp.asarray(cudf_zvec)
	del pd_zvec
	cudf_vec = cd.from_pandas(pd_vec)
	vec = cp.asarray(cudf_vec)
	del pd_vec
	xmins = x.unique()
	xmins = xmins.sort_values()
	xmins.index = cp.arange(0, xmins.shape[0])
	xmins = xmins.iloc[0: -1]
	if xmins.empty:
		print('(PLFIT) Error: x must contain at least two unique values.\n')
		alpha = 'Not a Number'
		xmin = x.iloc[0]
		D = 'Not a Number'
		return [alpha, xmin, D]
	xmax = x.max()
	z = x
	z = z.sort_values()
	z.index = cp.arange(0, z.shape[0])
	datA = -1 * cp.ones(xmins.shape[0])
	datB = -1 * cp.ones(xmins.shape[0])
	counter=0
	for xm in cp.arange(0, xmins.shape[0]):
		xmin = xmins.iloc[xm]
		xmin.index = 0
		tempz = z.loc[z >= xmin[0].astype(np.float)]
		n = tempz.shape[0]
		# estimate alpha via direct maximization of likelihood function
		logz = cp.log(tempz)
		slogz = cp.sum(logz)
		xminvec = cp.arange(1, xmin[0])
		L = -1 * n * cp.log(zvec[cp.newaxis, :] - cp.sum(cp.power(cp.arange(1, xmin[0])[:, cp.newaxis], -1*vec[cp.newaxis, :]),axis=0) ) - (vec * slogz) 
		I = L.argmax()
		# compute KS statistic
		fit = cp.cumsum(cp.power(cp.arange(xmin[0], xmax+1), -vec[I]) / (zvec[I] - cp.sum(cp.power(cp.arange(1, xmin[0]), -vec[I]))))
		#c = cp.arange(xmin[0], xmax + 1)
		#cdi = cp.sum(cp.less_equal(tempz.values[cp.newaxis, :], c[:, cp.newaxis]), axis=1) /  n
		cdi = cp.zeros(cp.arange(xmin.iloc[0].astype(np.int), xmax. astype(np.int) +1).shape[0])
		c = 0
		for XM in cp.arange(xmin.iloc[0].astype(np.int), xmax.astype(np.int) + 1):
			cdi[c] = cp.sum(cp.less_equal(cp.floor(tempz), XM)) / n
			c = c+1
		datA[xm] = cp.max(cp.abs(fit[0 : xmax - xmin[0] + 1] - cdi[0 : xmax - xmin[0] + 1]))
		datB[xm] = vec[I]
		
	I = cp.argmin(datA)
	xmin = xmins.iloc[I]
	xmin.index = 0
	z = x.loc[x.values >= xmin].values
	n = z.shape[0]
	alpha = datB[I]
	a = zvec[cp.argmax(vec[cp.less_equal(vec, alpha)])]
	b = cp.sum(cp.power(cp.arange(1, xmin[0]), -1*alpha))
	L = -alpha * cp.sum(cp.log(z)) - n * cp.log(a - b)
	return [alpha, xmin[0], L] 	
# helper functions
def _polyval(coeffs, x):
	p = coeffs.iloc[0]
	for c in coeffs.iloc[1:]:
		p = c + x * p
	return p

_zeta_int = pd.Series([ \
	-0.5,
	0.0,
	1.6449340668482264365, 1.2020569031595942854, 1.0823232337111381915,
	1.0369277551433699263, 1.0173430619844491397, 1.0083492773819228268,
	1.0040773561979443394, 1.0020083928260822144, 1.0009945751278180853,
	1.0004941886041194646, 1.0002460865533080483, 1.0001227133475784891,
	1.0000612481350587048, 1.0000305882363070205, 1.0000152822594086519,
	1.0000076371976378998, 1.0000038172932649998, 1.0000019082127165539,
	1.0000009539620338728, 1.0000004769329867878, 1.0000002384505027277,
	1.0000001192199259653, 1.0000000596081890513, 1.0000000298035035147,
	1.0000000149015548284])

_zeta_P = pd.Series([-3.50000000087575873, -0.701274355654678147,
	-0.0672313458590012612, -0.00398731457954257841,
	-0.000160948723019303141, -4.67633010038383371e-6,
	-1.02078104417700585e-7, -1.68030037095896287e-9,
	-1.85231868742346722e-11][::-1])

_zeta_Q = pd.Series([1.00000000000000000, -0.936552848762465319,
	-0.0588835413263763741, -0.00441498861482948666,
	-0.000143416758067432622, -5.10691659585090782e-6,
	-9.58813053268913799e-8, -1.72963791443181972e-9,
	-1.83527919681474132e-11][::-1])

_zeta_1 = pd.Series([3.03768838606128127e-10, -1.21924525236601262e-8,
	2.01201845887608893e-7, -1.53917240683468381e-6,
	-5.09890411005967954e-7, 0.000122464707271619326,
	-0.000905721539353130232, -0.00239315326074843037,
	0.084239750013159168, 0.418938517907442414, 0.500000001921884009])

_zeta_0 = pd.Series([-3.46092485016748794e-10, -6.42610089468292485e-9,
	1.76409071536679773e-7, -1.47141263991560698e-6, -6.38880222546167613e-7,
	0.000122641099800668209, -0.000905894913516772796, -0.00239303348507992713,
	0.0842396947501199816, 0.418938533204660256, 0.500000000000000052])


def zeta(s):
#	"""
#	Riemann zeta function, real argument
#	"""
#	
#	if not isinstance(s, (float, int)):
#		try:
#			s = float(s)
#		except (ValueError, TypeError):
#			try:
#				s = complex(s)
#				if not s.imag:
#					return complex(zeta(s.real))
#			except (ValueError, TypeError):
#				pass
#			raise NotImplementedError
	if s == 1:
		raise ValueError("zeta(1) pole")
	if s >= 27:
		return 1.0 + 2.0 ** (-s) + 3.0 ** (-s)
	n = int(s)
	if n == s:
		if n >= 0:
			return _zeta_int.iloc[n]
		if not (n % 2):
			return 0.0
	if s <= 0.0:
		return 0
	if s <= 2.0:
		if s <= 1.0:
			return _polyval(_zeta_0, s) / (s - 1)
		return _polyval(_zeta_1, s) / (s - 1)
	z = _polyval(_zeta_P, s) / _polyval(_zeta_Q, s)
	return 1.0 + 2.0 ** (-s) + 3.0 ** (-s) + 4.0 ** (-s) * z


