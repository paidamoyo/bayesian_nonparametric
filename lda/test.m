clear all;clc;
load '20news_train_bow';
load '20news_test_bow';
load '20news_train_labels';
load '20news_test_labels';
load '20news_words';
X1=X_train;
X2=X_test;
l1=labels_train;
l2=labels_test;

[m,n]=size(X1);
i=1;
while (i<=m)
    if nnz(X1(i,:))==0
        X1(i,:)=X1(m,:);
        l1(i)=l1(m);
        m=m-1;
    else
        i=i+1;
    end
end

[m1,n1]=size(X2);
i=1;
while (i<=m1)
    if nnz(X2(i,:))==0
        X2(i,:)=X2(m1,:);
        l2(i)=l2(m1);
        m1=m1-1;
    else
        i=i+1;
    end
end

i=1;
while (i<=n)
    if (nnz(X1(:,i))<5 || nnz(X1(:,i))>m/2) &&(nnz(X2(:,i))<5 || nnz(X2(:,i))>m1/2) 
        X1(:,i)=X1(:,n);
        X2(:,i)=X2(:,n);
        words(i)=words(n);
        n=n-1;
    else
        i=i+1;
    end
end
n1=n;
train_X=X1(1:m,1:n);
train_labels=l1(1:m);
test_X=X2(1:m1,1:n1);
test_labels=l2(1:m1);