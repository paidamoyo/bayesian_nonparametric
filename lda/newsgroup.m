clear all;clc;

load 'train_X' % train_X is N * P. N is the number of data points. P is the size of vocabulary
load 'test_X'  % test_X is M * P. M is the number of data points. P is the size of vocabulary
load 'train_labels'  % train_labels is 1 * N.
load 'test_labels' % test_labels is 1 * N.
topics = 20;
em_max_iter = 100;
vbe_max_iter=50;

%%%%%%%%%%%%%%%%%%%%%%%load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data.dicwordnum is the number of terms in the vocabulary:M/V.
% data.docnum is the number of documents:D.
% data.rate is response variable: y.
% data.doc.wordnum is the number of terms in each document:Nd.
% data.doc.word_id is index of words.
% data.doc.word is the times a word appears.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
traindata=loadnewsgroup(train_X,train_labels);
testdata =loadnewsgroup(test_X,test_labels);
time_loaddata=toc

% tic
% traindata=loadnewsgroup(X_train,labels_train);
% testdata =loadnewsgroup(X_test,labels_test);
% time_loaddata=toc;

tic;
model_sLDA=sLDA(traindata,topics,em_max_iter,vbe_max_iter);
time_sLDA=toc
% fprintf('\nTotal time cost for training = %s\n', rtime(time_sLDA));

tic;
[pre_rate,eval_result,model_test,y,accuracy,accuracy1] = sLDA_test(testdata,model_sLDA,vbe_max_iter);
time_sLDAtest=toc
