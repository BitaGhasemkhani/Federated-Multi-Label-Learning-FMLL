// Copyright Header - Federated Multi-Label Learning (FMLL)
// Copyright (C) 2024 Bita GHASEMKHANI

using System;

namespace FMLL
{
    class Program
    {     
        public static PerformanceMeasure Federated_Multi_Label_Learning(string f, int numLabels)
        {
            int folds = 10;

            PerformanceMeasure pm = new PerformanceMeasure();
            
            double accuracy = 0;
            double precision = 0;
            double recall = 0;
            double fScore = 0;
            double ROC = 0;
            double PRC = 0; 

            for (int i = 0; i < numLabels; i++)
            {
                java.util.Random rand = new java.util.Random(1);
                
                weka.core.Instances insts = ReadInstances(new java.io.FileReader("datasets\\" + f));
                
                insts.setClassIndex(0);

                for (int j = numLabels-1; j >= 0; j--)
                {
                    if (j > i)
                       insts.deleteAttributeAt(insts.numAttributes() - 1);
                    else if (j<i)
                       insts.deleteAttributeAt(insts.numAttributes() - 2);
                }

                insts.setClassIndex(insts.numAttributes() - 1);     

                weka.classifiers.trees.REPTree classifier = new weka.classifiers.trees.REPTree();

                weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(insts);
                eval.crossValidateModel(classifier, insts, folds, rand);

                accuracy = accuracy + eval.pctCorrect();
                precision = precision + eval.weightedPrecision();
                recall = recall + eval.weightedRecall();
                fScore = fScore + eval.weightedFMeasure();                 
                ROC = ROC + eval.weightedAreaUnderROC();
                PRC = PRC + eval.weightedAreaUnderPRC();
            }
                   
            pm.Accuracy = Math.Round(accuracy/numLabels, 2);
            pm.Precision = Math.Round(precision/numLabels, 3);
            pm.Recall = Math.Round(recall/numLabels, 3);
            pm.FScore = Math.Round(fScore/numLabels, 3);           
            pm.ROC = Math.Round(ROC / numLabels, 3);
            pm.PRC = Math.Round(PRC / numLabels, 3);
            return pm;
        }
  
        public static weka.core.Instances ReadInstances(java.io.FileReader fr)
        {
            weka.core.Instances instances = new weka.core.Instances(fr);
            instances.setClassIndex(instances.numAttributes() - 1);

            return instances;
        }      
                
        static void Main(string[] args)
        {
            string[] filenames = { "Amphibians", "Anuran-Calls", "HackerEarth-Adopt-A-Buddy"  };
            int[] numLabels = { 7, 3, 2 };

            Console.WriteLine("Federated Multi-Label Learning \n");
            Console.WriteLine("Dataset                     Accuracy     Precision  Recall       F-score    ROC         PRC");

            for (int i = 0; i < filenames.Length; i++)
            {
                Console.Write(filenames[i]);      
                PerformanceMeasure pm = Federated_Multi_Label_Learning(filenames[i] + ".arff", numLabels[i]);
                Console.SetCursorPosition(28, i + 3);
                Console.WriteLine( pm.Accuracy + "\t " + pm.Precision + "\t    " + pm.Recall + "\t " + pm.FScore + "\t    " + pm.ROC + "\t" + pm.PRC);         
            }
            Console.ReadLine();
        }
    }
}

