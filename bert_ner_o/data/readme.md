#### 本篇沒有提供資料， 可參考這邊的 [data](https://github.com/ProHiryu/bert-chinese-ner/tree/master/data)

準備好之後，調整這部分的輸入輸出即可

```
class NerProcessor_from_file(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "for_train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "for_dev_and_test.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "for_dev_and_test.txt")), "test")


    def get_labels(self):
        # prevent potential bug for chinese text mixed with english text
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
        # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
        global traindata_dir
        examples = self.get_train_examples(traindata_dir)
        tags = ["[CLS]","[SEP]"]
        for e in examples:
            
            for l in e.label.split(" "):
                if l not in tags:
                  tags.append(l)
        return tags
        #return ["o", "B-CHA", "I-CHA", "B-TIM", "I-TIM", "B-MON", "I-MON", "B-PER", "I-PER","[CLS]","[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples
```
