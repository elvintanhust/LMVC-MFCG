% % http://blog.csdn.net/google19890102/article/details/37656733
% % 
% % һ�������ܶȵľ����㷨�ĸ���
% %     �����Science�ϵ�һƪ�����ܶȵľ����㷨��Clustering by fast search and find of density peaks�������˴�ҵĹ�ע(���ҵĲ��ġ������еĻ���ѧϰ�㷨���������ܶȷ�ֵ�ľ����㷨����Ҳ���������ĵ�����)�������Ҿ����˽��»����ܶȵľ����㷨����Ϥ�»����ܶȵľ����㷨����ھ���ľ����㷨����K-Means�㷨֮�������
% %     �����ܶȵľ����㷨��Ҫ��Ŀ����Ѱ�ұ����ܶ��������ĸ��ܶ���������ھ���ľ����㷨��ͬ���ǣ����ھ���ľ����㷨�ľ���������״�Ĵأ��������ܶȵľ����㷨���Է���������״�ľ��࣬����ڴ��������������������Ҫ�����á�
% % ����DBSCAN�㷨��ԭ��
% % 1����������
% %     DBSCAN(Density-Based Spatial Clustering of Application with Noise)��һ�ֵ��͵Ļ����ܶȵľ����㷨����DBSCAN�㷨�н����ݵ��Ϊһ�����ࣺ
% % ���ĵ㡣�ڰ뾶Eps�ں��г���MinPts��Ŀ�ĵ�
% % �߽�㡣�ڰ뾶Eps�ڵ������С��MinPts���������ں��ĵ��������
% % �����㡣�Ȳ��Ǻ��ĵ�Ҳ���Ǳ߽��ĵ�
% % ����������������һ���ǰ뾶Eps����һ����ָ������ĿMinPts��
% %     һЩ�����ĸ���
% % Eps���򡣼������������ľ���С�ڵ���Eps�����еĵ�ļ��ϣ����Ա�ʾΪ��
% % ֱ���ܶȿɴ����ں��Ķ����Eps�����ڣ���ƶ���Ӷ��������ֱ���ܶȿɴ�ġ�
% % �ܶȿɴ���ڶ����������Ǵӹ���Eps��MinPtsֱ���ܶȿɴ�ģ�������ǴӶ������Eps��MinPts�ܶȿɴ�ġ�
% % 2���㷨����



% -------------------------------------------------------------------------
% Function: [class,type]=dbscan(x,k,Eps)
% -------------------------------------------------------------------------
% Aim: 
% Clustering the data with Density-Based Scan Algorithm with Noise (DBSCAN)
% -------------------------------------------------------------------------
% Input: 
% x - data set (m,n); m-objects, n-variables
% k - number of objects in a neighborhood of an object 
% (minimal number of objects considered as a cluster)
% Eps - neighborhood radius, if not known avoid this parameter or put []
% -------------------------------------------------------------------------
% Output: 
% class - vector specifying assignment of the i-th object to certain 
% cluster (m,1)
% type - vector specifying type of the i-th object 
% (core: 1, border: 0, outlier: -1)
% -------------------------------------------------------------------------
% Example of use:
% x=[randn(30,2)*.4;randn(40,2)*.5+ones(40,1)*[4 4]];
% [class,type]=dbscan(x,5,[])
% clusteringfigs('Dbscan',x,[1 2],class,type)
% -------------------------------------------------------------------------
% References:
% [1] M. Ester, H. Kriegel, J. Sander, X. Xu, A density-based algorithm for 
% discovering clusters in large spatial databases with noise, proc. 
% 2nd Int. Conf. on Knowledge Discovery and Data Mining, Portland, OR, 1996, 
% p. 226, available from: 
% www.dbs.informatik.uni-muenchen.de/cgi-bin/papers?query=--CO
% [2] M. Daszykowski, B. Walczak, D. L. Massart, Looking for 
% Natural Patterns in Data. Part 1: Density Based Approach, 
% Chemom. Intell. Lab. Syst. 56 (2001) 83-92 
% -------------------------------------------------------------------------
% Written by Michal Daszykowski
% Department of Chemometrics, Institute of Chemistry, 
% The University of Silesia
% December 2004
% http://www.chemometria.us.edu.pl


% -------------------------------------------------------------------------
% Input: 
% x - data set (m,n); m-objects, n-variables
% k - number of objects in a neighborhood of an object 
% (minimal number of objects considered as a cluster)
% Eps - neighborhood radius, if not known avoid this parameter or put []
% -------------------------------------------------------------------------


function [class,type]=dbscan(x,k,Eps)

[m,n]=size(x);  %�õ����ݵĴ�С

if nargin<3 | isempty(Eps)
   [Eps]=epsilon(x,k);                %  ??
end

x=[[1:m]' x];
[m,n]=size(x);      %���¼������ݼ��Ĵ�С
type=zeros(1,m);  %�������ֺ��ĵ�1���߽��0��������-1
no=1;                         %���ڱ����
touched=zeros(m,1);  %�����жϸõ��Ƿ�����,0��ʾδ������

%% ��ÿһ������д���
for i=1:m
    %�ҵ�δ�����ĵ�
    if touched(i)==0;
       ob=x(i,:);
       D=dist(ob(2:n),x(:,2:n));  %ȡ�õ�i���㵽�������е�ľ���
       ind=find(D<=Eps);            %�ҵ��뾶Eps�ڵ����е�
    
       %% ���ֵ������
       
       %�߽��
       if length(ind)>1 & length(ind)<k+1       
          type(i)=0;
          class(i)=0;
       end
       
       %������
       if length(ind)==1
          type(i)=-1;
          class(i)=-1;  
          touched(i)=1;
       end

       %���ĵ�(�˴��ǹؼ�����)
       if length(ind)>=k+1; 
          type(i)=1;
          class(ind)=ones(length(ind),1)*max(no);
          
          % �жϺ��ĵ��Ƿ��ܶȿɴ�
          while ~isempty(ind)
                ob=x(ind(1),:);
                touched(ind(1))=1;
                ind(1)=[];
                D=dist(ob(2:n),x(:,2:n));  %�ҵ���ind(1)֮��ľ���
                i1=find(D<=Eps);
     
                if length(i1)>1   %������������
                   class(i1)=no;
                   if length(i1)>=k+1;
                      type(ob(1))=1;
                   else
                      type(ob(1))=0;
                   end

                   for i=1:length(i1)
                       if touched(i1(i))==0
                          touched(i1(i))=1;
                          ind=[ind i1(i)];   
                          class(i1(i))=no;
                       end                    
                   end
                end
          end
          no=no+1; 
       end
   end
end
% ���������δ����ĵ�Ϊ������
i1=find(class==0);
class(i1)=-1;
type(i1)=-1;


%...........................................
function [Eps]=epsilon(x,k)

% Function: [Eps]=epsilon(x,k)
%
% Aim: 
% Analytical way of estimating neighborhood radius for DBSCAN
%
% Input: 
% x - data matrix (m,n); m-objects, n-variables
% k - number of objects in a neighborhood of an object
% (minimal number of objects considered as a cluster)



[m,n]=size(x);

Eps=((prod(max(x)-min(x))*k*gamma(.5*n+1))/(m*sqrt(pi.^n))).^(1/n);


%............................................
function [D]=dist(i,x)

% function: [D]=dist(i,x)
%
% Aim: 
% Calculates the Euclidean distances between the i-th object and all objects in x	 
%								    
% Input: 
% i - an object (1,n)
% x - data matrix (m,n); m-objects, n-variables	    
%                                                                 
% Output: 
% D - Euclidean distance (m,1)



[m,n]=size(x);
D=sqrt(sum((((ones(m,1)*i)-x).^2)'));

if n==1
   D=abs((ones(m,1)*i-x))';
end