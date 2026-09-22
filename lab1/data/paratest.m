try
    % 1. Задаем архитектуру простейшей нейросети
    layers = [
        imageInputLayer([28 28 1], 'Name', 'input')
        convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv')
        reluLayer('Name', 'relu')
        fullyConnectedLayer(10, 'Name', 'fc')
        softmaxLayer('Name', 'softmax')
    ];
    
    % 2. Проверяем сеть встроенным анализатором
    disp('--- Проверка структуры нейросети ---');
    analyzeNetwork(layers);
    
    % 3. Выводим сообщение об успешном тесте
    disp(' ');
    disp('🎉 Успех! Deep Learning Toolbox установлен и работает корректно.');
    
catch ME
    disp(' ');
    disp('❌ Ошибка! Похоже, Deep Learning Toolbox НЕ установлен.');
    disp(['Текст ошибки: ' ME.message]);
end

