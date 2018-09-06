
function [bhat p wh se ci t_stat]=robust_fit(x,y,ch)
%
% usage: [bhat p wh se ci t_stat]=robust_fit(x,y,ch)
%
%INPUT
%      x: independent variable/predictor/regressor. 
%      y: dependent variable.
%     ch: 1 - Tukey bi-square estimator (stronger penalty on median
%             deviation from least squares fit)
%         2 - Huber's estimator (more forgiving on median deviation from
%             least squares fit)
%OUTPUT
%      bhat: estimated regression coefficients (intercept and slope), with
%            plots of the estimated (via bootstrapping of residuals) sampling 
%            distribution for each parameter.
%         p: p-value/significance test from boostrapping the regression
%            residuals.
%        wh: Weighting of each observation 
%        se: Standard error of the coefficients' sampling distribution
%            (S.D of bootstrap regression estimates).
%        ci: 95% C.I of coefficients from the boostrap regression estimates
%            It is a 2X2 matrix, each row is the C.I for the
%            coefficients; first row is for the intercept.
%    t_stat: t-statistic associated with the coefficients
%
%         
%NOTES
%     The algorithm uses Robust estimators to correct for violations of
%     homoscedasticity and outliers, a common problem in regular datasets.
%     
%
%
%
%EXAMPLE 1. No relation between x and y
% x=(1:20)';
% y=randn(20,1);
% [bhat p wh se ci t_stat]=robust_fit(x,y,1);
% figure;plot(x,y,'.'); hold on; plot(x,bhat(1)+bhat(2).*x,'r')
% plot(x,[ones(length(x),1) x]*(([ones(length(x),1) x]'*[ones(length(x),1) x])\([ones(length(x),1) x]'*y)),'k')
% legend('Data','Robust','Non-robust')
% 
%
%EXAMPLE 2. -ve relation between x and y and no outlier
% x=(1:20)';
% y=-2-.3*x+randn(20,1);
% [bhat p wh se ci t_stat]=robust_fit(x,y,1);
% figure;plot(x,y,'.'); hold on; plot(x,bhat(1)+bhat(2).*x,'r')
% plot(x,[ones(length(x),1) x]*(([ones(length(x),1) x]'*[ones(length(x),1) x])\([ones(length(x),1) x]'*y)),'k')
% legend('Data','Robust','Non-robust')
% 
% 
%EXAMPLE 3. -ve relation between x and y with 20% outliers
% x=(1:20)';
% y=-2-.4*x+randn(20,1);
% I=randperm(length(y),4);
% y(I)=-y(I);
% [bhat p wh se ci t_stat]=robust_fit(x,y,1);
% figure;plot(x,y,'.'); hold on; plot(x,bhat(1)+bhat(2).*x,'r')
% plot(x,[ones(length(x),1) x]*(([ones(length(x),1) x]'*[ones(length(x),1) x])\([ones(length(x),1) x]'*y)),'k')
% legend('Data','Robust','Non-robust')
% 
%
%Example 4: no relation between x and y but 20% outliers gives spurious correlation
% x=(1:20)';
% y=randn(20,1);
% I=randperm(length(y),4);
% y(I)=y(I)+.4*x(I);
% [bhat p wh se ci t_stat]=robust_fit(x,y,1);
% figure;plot(x,y,'.'); hold on; plot(x,bhat(1)+bhat(2).*x,'r')
% plot(x,[ones(length(x),1) x]*(([ones(length(x),1) x]'*[ones(length(x),1) x])\([ones(length(x),1) x]'*y)),'k')
% legend('Data','Robust','Non-robust')
% 
%
%%%%%%% N.N @  May 2018 v1.0


% calling the Robust estimator

%x=x(:);
%y=y(:);
if ch==1
    [bhat1 wh1 x1]=tukey(x,y);
elseif ch==2
    [bhat1 wh1 x1]=huber(x,y);
end
bhat=bhat1;
wh=wh1;


% boostrapping of residuals to get standard error estimates of the
% regression coefficients, confidence intervals and p-values. Future work
% will seek to applying a weighting to the residuals similar in spirit to
% Matias Salibian-Barrera and Ruben H. Zamar (2002), Annals of Statistics.



yhat=x1*bhat;
e=(1).*(y-yhat);
%e=(wh).*(y-yhat);
e=e-mean(e);
e=e*((length(y)/(length(y)-2))^(0.5));
bhat_c=[];
rng default
parfor i=1:1000
    I=randi(length(e),1,length(e)); %sampling with replacement aka bootstrapping
    e1=e(I);
    y1=yhat+e1;
        if ch==1
            bhat_c(:,i) = tukey(x,y1);
        elseif ch==2
            bhat_c(:,i) = huber(x,y1);
        end
   % x11=[ones(length(x),1) x];
   % bhat_c(:,i)=(x11'*x11)\(x11'*y1);
    
end

% 95% C.I limits for the coefficients
ci=sort(bhat_c');
I=isnan(ci);
I=I(sum(I')==0);
ci=ci(logical(1-I),:);
ci=ci([floor(0.025*size(ci,1));ceil(0.975*size(ci,1))],:);
ci=ci';

% S.E of coefficients; This is the S.D of the generated sampling
% distribution
se=nanstd(bhat_c')';

% t-statistic
t_stat = bhat./se;

% p-values
p = 2*tcdf(abs(t_stat),length(x)-2,'upper');
% 
% disp(' ');
% disp('**** Output of robust_fit');
% disp(' ');
% name={'intercept';'slope'};
% Result=table(bhat,se,ci,t_stat,p,'RowNames',name)
% figure;hist(bhat_c(1,:),50);vline(bhat(1));title('Intercept and estimated sampling distribution')
% figure;hist(bhat_c(2,:),50);vline(bhat(2));title('Slope and estimated sampling distribution')
end


function [bhat w x1]=huber(x,y)
% setting things up
x1=[ones(length(x),1) x];
w=ones(length(y),1);
W=diag(w);
% first run to get robust measure of residual error after least squares
% fit
bhat=(x1'*W*x1)\(x1'*W*y); 
yhat=x1*bhat;
r=y-yhat;
g=zeros(length(w),1);
m1=mad(r,1); % median absolute deviation
w=min(1,(0.6745*2*m1/0.6745)./abs(r)); % Huber's weight function with 95% efficiency against outliers.
tol=1;
chk=1;

% iterative re-weighted least squares
while tol > 1e-10
    W=diag(w);
    bhat=(x1'*W*x1)\(x1'*W*y); % weighted least squares
    yhat=x1*bhat;
    r=y-yhat;
    w=min(1,(0.6745*2*m1/0.6745)./abs(r)); % Huber weights
    tol=norm(g-w);
    g=w;
    chk=chk+1;
    if chk==1000
        break
    end
end

warning('off')
end


function [bhat w x1]=tukey(x,y)
% setting things up
x1=[ones(length(x),1) x];
w=ones(length(y),1);
W=diag(w);
% first run to get robust measure of residual error after least squares
% fit
bhat=(x1'*W*x1)\(x1'*W*y);
yhat=x1*bhat;
r=y-yhat;
g=zeros(length(w),1);
m1=mad(r,1); % median absolute deviation
k=(4.685*m1/0.6745); % Tukey tuning constant 95% efficiency
w=(1-((r/k).^2)).^2; % Tukey bisquare weights
w(abs(r)>k)=0;
tol=1;
chk=1;

% iterative re-weighted least squares
while tol > 1e-10
    W=diag(w);
    bhat=(x1'*W*x1)\(x1'*W*y);  % weighted least squares
    yhat=x1*bhat;
    r=y-yhat;
    w=(1-((r/k).^2)).^2; % Tukey bisquare weights
    w(abs(r)>k)=0;
    tol=norm(g-w);
    g=w;
    chk=chk+1;
    if chk==1000
        break
    end
end

warning('off')
end






