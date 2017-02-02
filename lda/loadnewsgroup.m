function corp = loadnewsgroup(X,labels)

[m,n]=size(X);% m documents and n words
%remove stop words and no words document
% i=1;
% while (i<=m)
%     if nnz(X1(i,:))==0
%         X1(i,:)=X1(m,:);
%         label(i)=label(m);
%         m=m-1;
%     else
%         i=i+1;
%     end
% end
% i=1;
% while (i<=n)
%     if nnz(X1(:,i))<5 || nnz(X1(:,i))>m/2
%         X1(:,i)=X1(:,n);
%         n=n-1;
%     else
%         i=i+1;
%     end
% end
% 
% X=X1(1:m,1:n);
% labels=label(1:m);

totalwords=0;
for i=1:m
    corp.doc(i).id=i;
    corp.doc(i).rate=labels(i);
    corp.rate(i)=corp.doc(i).rate;
    corp.doc(i).docwordnum=0;
    [number,corp.doc(i).word_id,corp.doc(i).word]=find(X(i,:));
    corp.doc(i).docwordnum=sum(corp.doc(i).word);
    totalwords=totalwords+ corp.doc(i).docwordnum;
end

corp.docnum=m;
corp.dicwordnum=n;
corp.totalwords=totalwords;

