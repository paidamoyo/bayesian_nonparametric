% function [pre_rate, eval_result, model_test] = sLDA_test(testdata, model, vbe_max_iter);

model_test.gammas = repmat(0.0, testdata.docnum, model.K);
for i=1:testdata.docnum
    gamma= model.alpha + repmat(testdata.doc(i).docwordnum/model.K, 1, model.K);
    model_test.gammas(i,:) =gamma;
end
model_test.phi=repmat(1/model.K, testdata.dicwordnum, model.K);

dict = find(sum(model.beta, 1)~=0);

for i=1:376%testdata.docnum,
    % Remove words not occur train data.
    [comid, idx_src, idx_tar] = intersect(testdata.doc(i).word_id, dict);
    testdata.doc(i).word_id = testdata.doc(i).word_id(idx_src);
    testdata.doc(i).word = testdata.doc(i).word(idx_src);
    
    for iter=1:vbe_max_iter,
    % update phi
        model_test.phi(testdata.doc(i).word_id,:) = model.beta(:,testdata.doc(i).word_id)'*diag(exp(psi(model_test.gammas(i,:))));
        nm_const = sum(model_test.phi(testdata.doc(i).word_id,:), 2);
        model_test.phi(testdata.doc(i).word_id,:)= diag(1./nm_const)*model_test.phi(testdata.doc(i).word_id,:);

    % update gamma
        model_test.gammas(i,:) = model.alpha + testdata.doc(i).word*model_test.phi(testdata.doc(i).word_id,:);
    end
    
    aver_beta = sum(diag(testdata.doc(i).word)*model_test.phi(testdata.doc(i).word_id, :))...
        ./testdata.doc(i).docwordnum;
    pre_rate(i) = aver_beta * model.eta;
end
mean_y=sum(testdata.rate)/testdata.docnum;
eval_result=1-sum((testdata.rate-pre_rate).^2)/sum((testdata.rate-mean_y).^2);
fprintf(1, 'The result of predictive R2 for sLDA is %f\n', eval_result);