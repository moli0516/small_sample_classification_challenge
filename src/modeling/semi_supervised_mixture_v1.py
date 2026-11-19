"""Weighted semi-supervised Gaussian/Bernoulli mixture; no file I/O."""
import warnings
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture


class SemiSupervisedMixture:
    def __init__(self,n_components=4,covariance_type="diag",unlabelled_weight=1.,
                 reg_covar=1e-3,label_smoothing=1e-3,max_iter=300,tol=1e-4,random_state=42):
        self.n_components=n_components;self.covariance_type=covariance_type
        self.unlabelled_weight=unlabelled_weight;self.reg_covar=reg_covar
        self.label_smoothing=label_smoothing;self.max_iter=max_iter;self.tol=tol;self.random_state=random_state

    def _log_density(self,x):
        d=x.shape[1];ans=[]
        for k in range(self.n_components):
            diff=x-self.means_[k]
            if self.covariance_type=="diag":
                cov=self.covariances_[k]
                ans.append(-.5*(d*np.log(2*np.pi)+np.log(cov).sum()+((diff**2)/cov).sum(axis=1)))
            else:
                chol=np.linalg.cholesky(self.covariances_)
                z=solve_triangular(chol,diff.T,lower=True,check_finite=False)
                ans.append(-.5*(d*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+(z*z).sum(axis=0)))
        return np.column_stack(ans)+np.log(self.weights_)

    def _terms(self,x,y,n_labelled):
        logp=self._log_density(x)
        logp[:n_labelled]+=np.where(y[:,None]==1,np.log(self.class_probabilities_),np.log1p(-self.class_probabilities_))
        norms=logsumexp(logp,axis=1)
        return np.exp(logp-norms[:,None]),norms

    def fit(self,x_labelled,y,x_unlabelled=None):
        xl=np.asarray(x_labelled,dtype=float);y=np.asarray(y)
        if xl.ndim!=2 or len(xl)!=len(y) or set(np.unique(y))!={0,1}:
            raise ValueError("Finite labelled matrix and both binary classes required")
        xu=np.empty((0,xl.shape[1])) if x_unlabelled is None else np.asarray(x_unlabelled,dtype=float)
        if xu.ndim!=2 or xu.shape[1]!=xl.shape[1]:raise ValueError("Unlabelled dimensions differ")
        x=np.vstack([xl,xu]);nl=len(xl)
        if not np.isfinite(x).all() or self.covariance_type not in ("diag","tied"):
            raise ValueError("Invalid data/covariance")
        if not 0<self.n_components<=len(x) or self.unlabelled_weight<0 or self.reg_covar<=0 or self.label_smoothing<=0 or self.max_iter<1 or self.tol<=0:
            raise ValueError("Invalid mixture settings")
        a=np.r_[np.full(nl,1/nl),np.full(len(xu),self.unlabelled_weight/max(1,len(xu)))]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            init=GaussianMixture(n_components=self.n_components,covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,n_init=1,max_iter=100,random_state=self.random_state).fit(x)
        self.initialization_warnings_=[str(w.message) for w in caught]
        self.weights_=init.weights_.copy();self.means_=init.means_.copy();self.covariances_=init.covariances_.copy()
        r=init.predict_proba(xl);s=self.label_smoothing
        self.class_probabilities_=np.clip(((r*y[:,None]).sum(axis=0)/nl+s)/(r.sum(axis=0)/nl+2*s),1e-8,1-1e-8)
        self.converged_=False;self.objective_trace_=[]
        _,norms=self._terms(x,y,nl)
        objective=lambda v:float(a@v+s*(np.log(self.class_probabilities_)+np.log1p(-self.class_probabilities_)).sum())
        previous=objective(norms);self.objective_trace_.append(previous)
        for iteration in range(1,self.max_iter+1):
            r,_=self._terms(x,y,nl);w=r*a[:,None];mass=np.maximum(w.sum(axis=0),1e-12)
            self.weights_=mass/mass.sum();self.means_=(w.T@x)/mass[:,None]
            if self.covariance_type=="diag":
                self.covariances_=np.array([(w[:,k,None]*(x-self.means_[k])**2).sum(axis=0)/mass[k]+self.reg_covar for k in range(self.n_components)])
            else:
                covariance=np.zeros((x.shape[1],x.shape[1]))
                for k in range(self.n_components):
                    diff=x-self.means_[k];covariance+=(diff*w[:,k,None]).T@diff
                self.covariances_=covariance/mass.sum()+self.reg_covar*np.eye(x.shape[1])
            labelled=r[:nl]/nl
            self.class_probabilities_=np.clip(((labelled*y[:,None]).sum(axis=0)+s)/(labelled.sum(axis=0)+2*s),1e-8,1-1e-8)
            _,norms=self._terms(x,y,nl);current=objective(norms)
            if not np.isfinite(current):raise FloatingPointError("Nonfinite objective")
            self.objective_trace_.append(current)
            delta=current-previous
            # Ridge covariance regularization is not exact maximization of an explicit covariance prior.
            # A material decrease is NOT called convergence.
            if abs(delta)<=self.tol*(1+abs(previous)):
                self.converged_=True;break
            previous=current
        self.n_iter_=iteration;self.effective_masses_=mass.tolist()
        self.objective_decreases_=int(np.sum(np.diff(self.objective_trace_) < -1e-8))
        return self

    def predict_proba(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=self.means_.shape[1] or not np.isfinite(x).all():raise ValueError("Invalid prediction matrix")
        logp=self._log_density(x);r=np.exp(logp-logsumexp(logp,axis=1)[:,None])
        p=np.clip(r@self.class_probabilities_,0,1)
        return np.column_stack([1-p,p])
