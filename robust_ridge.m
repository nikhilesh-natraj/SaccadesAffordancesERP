function [bhat edf r r2 w ]=robust_ridge(x1,y1,k)
%
% USAGE: [bhat edf r r2 w ]=robust_ridge(x1,y1,k,ch)
%
% INPUT:
%        x1 - design matrix (rows-observations, cols-independent variable/predictors)
%        y1 - Dependent variable (rows-observations)
%        k  - lambda, the ridge parameter
% OUTPUT
%      bhat - estimated betas or weights of the robust ridge model
%       edf - effective degrees of freedom of the robust ridge model
%         r - vector of regression residuals
%        r2 - model r-squared
%         w - Huber's weighting coefficients
% EXAMPLE
% y1=sin(2*pi*(20/100)*(1:100))'+randn(100,1); %20Hz sine sampled at 100Hz in noise
% x1=randn(100,1000); % 1000 predictors with only 100 data points
% x1(:,20) = y1*6.9 + randn(100,1); % predictor 20 is important
% x1(:,200) = x1(:,20)*3.5 + randn(100,1); % correlated predictor 200 is important
% y1(20:25) = y1(20:25) + 8; % wild outliers in 5% of data
% [bhat edf r r2 w ]=robust_ridge(x1,y1,0.01);
% figure;stem(bhat);title('Coefficients')
% figure;stem(w);title('Weighting for each of the 100 data points')
%
% N.N 2018


% setting things up
x1=zscore(x1);
y1=zscore(y1);
w=ones(length(y1),1);
W=diag(w);

% run robust ridge once to get residuals
bhat = (x1'*W*x1 + k*eye(size(x1,2)))\(x1'*W*y1);
yhat=x1*bhat;
r=y1-yhat;
g=zeros(length(w),1);
% median error
m1=mad(r,1);
tol=1;
chk=1;
% first init. of Huber
w=min(1,(1.345*m1/0.6745)./abs(r));

% iterative re-weighted least squares
while tol > 1e-6
    
    % Huber's robust ridge regression
    W=diag(w);    
    bhat = (x1'*W*x1 + k*eye(size(x1,2)))\(x1'*W*y1);
    yhat=x1*bhat;
    r=y1-yhat; %residuals 
    h= x1*((x1'*W*x1 + k*eye(size(x1,2)))\(x1'*W));  % hat matrix  
    w=min(1,(1.345*m1/0.6745)./abs(r));
    
    % tolerance check
    tol=norm(g-w);
    g=w;
    
    % regression diagnostics
    edf=trace(h);% leverages
    r2=(yhat'*W*yhat)/(y1'*W*y1); %r-squared
    
    % limit on number of iterations
    chk=chk+1;    
    if chk==750
        break
    end
end
