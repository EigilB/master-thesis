function [Mskekur] = Mskekur(X,c,alpha)
%Mardia's multivariate skewness and kurtosis.
%Calculates the Mardia's multivariate skewness and kurtosis coefficients
%as well as their corresponding statistical test. For large sample size 
%the multivariate skewness is asymptotically distributed as a Chi-square 
%random variable; here it is corrected for small sample size. However,
%both uncorrected and corrected skewness statistic are presented. Likewise,
%the multivariate kurtosis it is distributed as a unit-normal.
%
% Syntax: function [Mskekur] = Mskekur(X,c,alpha) 
%      
% Inputs:
%      X - multivariate data matrix [Size of matrix must be n(data)-by-p(variables)]. 
%      c - normalizes covariance matrix by n (c=1[default]) or by n-1 (c~=1)
%  alpha - significance level (default = 0.05). 
%
% Outputs:
%      -Complete statistical analysis table of both multivariate
%       Mardia's skewness and kurtosis.
%      -Chi-square quantile-quantile (Q-Q) plot of the squared Mahalanobis
%       distances of the observations from the mean vector.
%      -The file ask you whether or not are you interested to label the n
%       data points on the Q-Q plot:
%          Are you interested to explore all the n data points? (y/n):

%
%    Example:For the example of Pope et al. (1980) given by Stevens (1992, p. 249), 
%            with 12 cases (n = 12) and three variables (p = 3). We are interested
%            to calculate and testing its multivariate skewnees and kurtosis with a
%            covariance matrix centered by n and a significance level = 0.05 (default).
%                      --------------    --------------
%                       x1   x2   x3      x1   x2   x3
%                      --------------    --------------
%                      2.4  2.1  2.4     4.5  4.9  5.7
%                      3.5  1.8  3.9     3.9  4.7  4.7
%                      6.7  3.6  5.9     4.0  3.6  2.9
%                      5.3  3.3  6.1     5.7  5.5  6.2
%                      5.2  4.1  6.4     2.4  2.9  3.2
%                      3.2  2.7  4.0     2.7  2.6  4.1
%                      --------------    --------------
%
%  Total data matrix must be:
%  X=[2.4 2.1 2.4;3.5 1.8 3.9;6.7 3.6 5.9;5.3 3.3 6.1;5.2 4.1 6.4;
%  3.2 2.7 4.0;4.5 4.9 5.7;3.9 4.7 4.7;4.0 3.6 2.9;5.7 5.5 6.2;2.4 2.9 3.2;
%  2.7 2.6 4.1];
%
%  Calling on Matlab the function: 
%         Mskekur(X,1)
%
%  Answer is:
%
% Analysis of the Mardia's multivariate asymmetry skewness and kurtosis.
% [No. of data = 20, Variables = 4]
% ----------------------------------------------------------------------------
% Multivariate                    Coefficient      Statistic     df       P
% ----------------------------------------------------------------------------
% Skewness                          2.2927           4.5854      10    0.9171
% Skewness
% corrected for small sample        2.2927           6.4794      10    0.7735
% Kurtosis                         11.4775          -1.1139            0.2653
% ----------------------------------------------------------------------------
% With a given significance level of: 0.05
% The multivariate skewness results not significative.
% The multivariate skewness corrected for small sample results not significative.
% The multivariate kurtosis results significative.
%
% Are you interested to get the object labels? (y/n): y
%
% Note: At the end of the program execution. On the generated figures you can turn-on active button
% 'Edit Plot'(fifth icon from left to right:white arrow), do click on the selected label and drag
% it to fix it on the desired position. Then turn-off active 'Edit Plot'.
%
%  Created by A. Trujillo-Ortiz and R. Hernandez-Walls
%         Facultad de Ciencias Marinas        
%         Universidad Autonoma de Baja California 
%         Apdo. Postal 453  
%         Ensenada, Baja California
%         Mexico  
%         atrujo@uabc.mx
%
%  Copyright.May 22, 2003.
%
%  Version 2.0. Copyright.September 30, 2007
%  -- This new version normalizes covariance matrix by n or n-1 (n=number of observations).
%   The previous version 1.0, normalized covariance matrix only by n-1 --
%
%  To cite this file, this would be an appropriate format:
%  Trujillo-Ortiz, A. and R. Hernandez-Walls. (2003). Mskekur: Mardia's multivariate skewness
%    and kurtosis coefficients and its hypotheses testing. A MATLAB file. [WWW document]. URL 
%    http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=3519
%
%  References:
%  Mardia, K. V. (1970), Measures of multivariate skewnees and kurtosis with
%         applications. Biometrika, 57(3):519-530.
%  Mardia, K. V. (1974), Applications of some measures of multivariate skewness
%         and kurtosis for testing normality and robustness studies. Sankhyâ A,
%         36:115-128
%  Stevens, J. (1992), Applied Multivariate Statistics for Social Sciences. 2nd. ed.
%         New-Jersey:Lawrance Erlbaum Associates Publishers. pp. 247-248.
%

if nargin < 3, 
    alpha = 0.05;  %(default)
end 

if nargin < 2, 
    error('Requires at least two input arguments.'); 
end 

[n,p] = size(X);

difT = [];
for	j = 1:p
   eval(['difT=[difT,(X(:,j)-mean(X(:,j)))];']);
end

if c == 1  %covariance matrix normalizes by (n) [=default]
    S = cov(X,1);
else   %covariance matrix normalizes by (n-1)
    S = cov(X);
end

D = difT*inv(S)*difT';  %squared-Mahalanobis' distances matrix
b1p = (sum(sum(D.^3)))/n^2;  %multivariate skewness coefficient
b2p=trace(D.^2)/n;  %multivariate kurtosis coefficient

k = ((p+1)*(n+1)*(n+3))/(n*(((n+1)*(p+1))-6));  %small sample correction
v = (p*(p+1)*(p+2))/6;  %degrees of freedom
g1c = (n*b1p*k)/6;  %skewness test statistic corrected for small sample:it approximates to a chi-square distribution
g1 = (n*b1p)/6;  %skewness test statistic:it approximates to a chi-square distribution
P1 = 1 - chi2cdf(g1,v);  %significance value associated to the skewness
P1c = 1 - chi2cdf(g1c,v);  %significance value associated to the skewness corrected for small sample

g2 = (b2p-(p*(p+2)))/(sqrt((8*p*(p+2))/n));  %kurtosis test statistic:it approximates to
                                             %a unit-normal distribution
P2 = 2*(1-normcdf(abs(g2)));  %significance value associated to the kurtosis

disp(' ');
disp('Analysis of the Mardia''s multivariate asymmetry skewness and kurtosis.')
disp(['[No. of data = ',num2str(n) ', ' 'Variables = ',num2str(p) ']']);
%fprintf('No. of data = %i\n', n ';''Variables = %i\n', p');
fprintf('----------------------------------------------------------------------------\n');
disp('Multivariate                    Coefficient      Statistic      df      P')
fprintf('----------------------------------------------------------------------------\n');
fprintf('Skewness         %24.4f%17.4f%8i%10.4f\n\n',b1p,g1,v,P1);
disp('Skewness');
fprintf('corrected for small sample        %3.4f%17.4f%8i%10.4f\n\n',b1p,g1c,v,P1c);
fprintf('Kurtosis         %24.4f%17.4f%18.4f\n',b2p,g2,P2);
fprintf('----------------------------------------------------------------------------\n');
fprintf('With a given significance level of: %.2f\n', alpha);
if P1 >= alpha;
   fprintf('The multivariate skewness results not significative.\n');
else 
   fprintf('The multivariate skewness results significative.\n');
end

if P1c >= alpha;
   fprintf('The multivariate skewness corrected for small sample results not significative.\n');
else 
   fprintf('The multivariate skewness corrected for small sample results significative.\n');
end

if P2 >= alpha;
   fprintf('The multivariate kurtosis results not significative.\n\n');
else 
   fprintf('The multivariate kurtosis results significative.\n\n');
end

%Chi-square quantile-quantile (Q-Q) plot of the squared Mahalanobis
%distances of the observations from the mean vector.
[d,t] = sort(diag(D));   %squared Mahalanobis distances
r = tiedrank(d);  %ranks of the squared Mahalanobis distances
lb = input('Are you interested to get the object labels? (y/n): ','s');
if lb == 'y'
    figure;
    labels = strread(sprintf('%d ',t),'%s').';
    disp(' ')
    disp('Note: At the end of the program execution. On the generated figures you can turn-on active button')
    disp('''Edit Plot''(fifth icon from left to right:white arrow), do click on the selected label and drag')
    disp('it to fix it on the desired position. Then turn-off active ''Edit Plot''.')
    disp(' ')
    chi2q=chi2inv((r-0.5)./n,p);  %chi-square quantiles  
    plot(chi2q,d,'*b')
    text(chi2q,d,labels)
    axis([0 max(chi2q)+1 0 max(d)+1])
    xlabel('Chi-square quantile')
    ylabel('Squared Mahalanobis distance')
    title ('Chi-square Q-Q plot')
else
    chi2q=chi2inv((r-0.5)./n,p);  %chi-square quantiles  
    plot(chi2q,d,'*b')
    axis([0 max(chi2q)+1 0 max(d)+1])
    xlabel('Chi-square quantile')
    ylabel('Squared Mahalanobis distance')
    title ('Chi-square Q-Q plot')  
end

return,