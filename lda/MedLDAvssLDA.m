clear all;
clc;

topics = 20;
em_max_iter = 30;
vbe_max_iter=50;

%%%%%%%%%%%%%%%%%%%%%%%load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data.dicwordnum is the number of terms in the vocabulary:M/V.
% data.docnum is the number of documents:D.
% data.rate is response variable: y.
% data.doc.wordnum is the number of terms in each document:Nd.
% data.doc.word_id is index of words.
% data.doc.word is the times a word appears.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
traindata = loaddata('train_review.dat'); 
testdata = loaddata('test_review.dat');
time_loaddata=toc;

tic;
model_sLDA=sLDA(traindata,topics,em_max_iter,vbe_max_iter);
time_sLDA=toc;
% fprintf('\nTotal time cost for training = %s\n', rtime(time_sLDA));

tic;
[pre_rate,eval_result,model_test] = sLDA_test(testdata,model_sLDA,vbe_max_iter);
time_sLDAtest=toc;
