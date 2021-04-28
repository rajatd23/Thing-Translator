package com.example.thingtranslator;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;

public class Classify extends AppCompatActivity {
    TextToSpeech textToSpeech;

    // presets for rgb conversion
    private static final int RESULTS_TO_SHOW = 3;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    // options for model interpreter
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    // tflite graph
    private Interpreter tflite;
    // holds all the possible labels for model
    private List<String> labelList;
    // holds the selected image data as bytes
    private ByteBuffer imgData = null;
    // holds the probabilities of each label for non-quantized graphs
    private float[][] labelProbArray = null;
    // holds the probabilities of each label for quantized graphs
    private byte[][] labelProbArrayB = null;
    // array that holds the labels with the highest probabilities
    private String[] topLables = null;
    // array that holds the highest probabilities
    private String[] topConfidence = null;


    // selected classifier information received from extras
    private String chosen;
    private boolean quant;

    // input image dimensions for the Inception Model
    private int DIM_IMG_SIZE_X = 224;
    private int DIM_IMG_SIZE_Y = 224;
    private int DIM_PIXEL_SIZE = 3;

    // int array to hold image data
    private int[] intValues;
    private String[] Eng;
    private String[] Ger;
    private String[] Jap;
    private String[] Chi;
    private String[] Ita;
    private String[] Kor;
    private String[] Fre;
    int z = 0;
    // activity elements
    private ImageView selected_image;
    private Button classify_button;
    private Button back_button;
    private TextView label1;
    private TextView label2;
    private TextView label3;
    private TextView Confidence1;
    private TextView Confidence2;
    private TextView Confidence3;
    String lang;
    String str1;

    // priority queue that will hold the top results from the CNN
    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    private void speaker(String str1){
        Log.d("Test",str1);
//        Toast.makeText(this, str1, Toast.LENGTH_SHORT).show();
//        InitialPage initialPage =new InitialPage();
//        String str1=initialPage.lang.toString();
//        System.out.println(str1);
//        Toast.makeText(Classify.this,str1,Toast.LENGTH_LONG).show();
//        str = "GERMAN";
        if(str1.equals("GERMAN")){
            textToSpeech = new TextToSpeech(Classify.this, new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int i) {
                    if (i == TextToSpeech.SUCCESS){
                        textToSpeech.setLanguage(Locale.GERMAN);
                    }
                }
            });
        }
        else if(str1.equals("ENGLISH")){
            textToSpeech = new TextToSpeech(Classify.this, new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int i) {
                    if (i == TextToSpeech.SUCCESS){
                        textToSpeech.setLanguage(Locale.ENGLISH);
                    }
                }
            });
        }
        else if(str1.equals("JAPANESE")){
            textToSpeech = new TextToSpeech(Classify.this, new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int i) {
                    if (i == TextToSpeech.SUCCESS){
                        textToSpeech.setLanguage(Locale.JAPANESE);
                    }
                }
            });
        }
        else if(str1.equals("CHINESE")){
            textToSpeech = new TextToSpeech(Classify.this, new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int i) {
                    if (i == TextToSpeech.SUCCESS){
                        textToSpeech.setLanguage(Locale.CHINESE);
                    }
                }
            });
        }
        else if(str1.equals("ITALIAN")){
            textToSpeech = new TextToSpeech(Classify.this, new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int i) {
                    if (i == TextToSpeech.SUCCESS){
                        textToSpeech.setLanguage(Locale.ITALIAN);
                    }
                }
            });
        }
        else if(str1.equals("KOREAN")){
            textToSpeech = new TextToSpeech(Classify.this, new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int i) {
                    if (i == TextToSpeech.SUCCESS){
                        textToSpeech.setLanguage(Locale.KOREAN);
                    }
                }
            });
        }
        else {
            textToSpeech = new TextToSpeech(Classify.this, new TextToSpeech.OnInitListener() {
                @Override
                public void onInit(int i) {
                    if (i == TextToSpeech.SUCCESS){
                        textToSpeech.setLanguage(Locale.FRENCH);
                    }
                }
            });
        }

    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // get all selected classifier data from classifiers
        chosen = getIntent().getStringExtra("chosen");
        quant = getIntent().getBooleanExtra("quant", false);
        str1 = getIntent().getStringExtra("language");

        // initialize array that holds image data
        intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

        super.onCreate(savedInstanceState);

        speaker(str1);

        //initilize graph and labels
        try{
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
            labelList = loadLabelList();
        } catch (Exception ex){
            ex.printStackTrace();
        }

        // initialize byte array. The size depends if the input data needs to be quantized or not
        if(quant){
            imgData =
                    ByteBuffer.allocateDirect(
                            DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        } else {
            imgData =
                    ByteBuffer.allocateDirect(
                            4 * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        }
        imgData.order(ByteOrder.nativeOrder());

        // initialize probabilities array. The datatypes that array holds depends if the input data needs to be quantized or not
        if(quant){
            labelProbArrayB= new byte[1][labelList.size()];
        } else {
            labelProbArray = new float[1][labelList.size()];
        }

        setContentView(R.layout.activity_classify);

        // labels that hold top three results of CNN
        label1 = (TextView) findViewById(R.id.label1);
        label2 = (TextView) findViewById(R.id.label2);
        label3 = (TextView) findViewById(R.id.label3);
        // displays the probabilities of top labels
        Confidence1 = (TextView) findViewById(R.id.Confidence1);
        Confidence2 = (TextView) findViewById(R.id.Confidence2);
        Confidence3 = (TextView) findViewById(R.id.Confidence3);
        // initialize imageView that displays selected image to the user
        selected_image = (ImageView) findViewById(R.id.selected_image);

        // initialize array to hold top labels
        topLables = new String[RESULTS_TO_SHOW];
        // initialize array to hold top probabilities
        topConfidence = new String[RESULTS_TO_SHOW];

        // allows user to go back to activity to select a different image
        back_button = (Button)findViewById(R.id.back_button);
        back_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(Classify.this, SelectLanguage.class);
                startActivity(i);
            }
        });

        // classify current dispalyed image
        classify_button = (Button)findViewById(R.id.classify_image);
        classify_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // get current bitmap from imageView
                Bitmap bitmap_orig = ((BitmapDrawable)selected_image.getDrawable()).getBitmap();
                // resize the bitmap to the required input size to the CNN
                Bitmap bitmap = getResizedBitmap(bitmap_orig, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y);
                // convert bitmap to byte array
                convertBitmapToByteBuffer(bitmap);
                // pass byte data to the graph
                if(quant){
                    tflite.run(imgData, labelProbArrayB);
                } else {
                    tflite.run(imgData, labelProbArray);
                }
                // display the results
                printTopKLabels();
            }
        });

        // get image from previous activity to show in the imageView
        Uri uri = (Uri)getIntent().getParcelableExtra("resID_uri");
        try {
            Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
            selected_image.setImageBitmap(bitmap);
            // not sure why this happens, but without this the image appears on its side
            selected_image.setRotation(selected_image.getRotation() + 90);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // loads tflite grapg from file
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd(chosen);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // converts bitmap to byte array which is passed in the tflite graph
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                // get rgb values from intValues where each int holds the rgb values for a pixel.
                // if quantized, convert each rgb value to a byte, otherwise to a float
                if(quant){
                    imgData.put((byte) ((val >> 16) & 0xFF));
                    imgData.put((byte) ((val >> 8) & 0xFF));
                    imgData.put((byte) (val & 0xFF));
                } else {
                    imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                }

            }
        }
    }

    // loads the labels from the label txt file in assets into a string array
    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open("tera.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // print the top labels and respective confidences
    private void printTopKLabels() {
        // add all results to priority queue
        for (int i = 0; i < labelList.size(); ++i) {
            if(quant){
                sortedLabels.add(
                        new AbstractMap.SimpleEntry<>(labelList.get(i), (labelProbArrayB[0][i] & 0xff) / 255.0f));
            } else {
                sortedLabels.add(
                        new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
            }
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }

        // get top results from priority queue
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            topLables[i] = label.getKey();
            topConfidence[i] = String.format("%.0f%%",label.getValue()*100);
        }

        // set the corresponding textviews with the results
        label1.setText("1. "+topLables[2]);
        label2.setText("2. "+topLables[1]);
        label3.setText("3. "+topLables[0]);
        Confidence1.setText(topConfidence[2]);
        Confidence2.setText(topConfidence[1]);
        Confidence3.setText(topConfidence[0]);
        Eng = new String[]{"air conditioner", "Airpods", "badminton racket", "badminton shuttle", "Bandages", "baseball bat", "bean bag", "belt", "bicycle", "bottle", "Bow Tie", "box", "Bulb", "Calender", "Ceiling fan", "Cell battery", "chair", "charger", "clock", "comb", "computer keyboard", "computer mouse", "Computer Ram", "CPU", "cricket bat", "curtains", "Cushion", "daisy", "dandelion", "Digital Camera", "Digital Scale", "Diode", "dogs", "door", "door mat", "dustbin", "DVD Disc", "dvd player", "Extension Cords", "Face Mask", "Fan capacitor", "Floppy Disk", "Fork"," Formal Shirt", "Gas Cylinder","Jeans", "knife", "laptop", "Memory Card", "microwave"," Motherboard", "Mugs", "Neck Tie"," pen", "Pencil", "Razor", "refrigerator", "Relay Circuit", "roses", "Scale", "scissor", "shoes", "Sim Card", "Socks", "Spectacles", "spoon", "sunflower"," Table fan"," table lamp", "television", "toothbrush", "Transistor", "Trimmer", "Trophy"," T-Shirt", "Tubelight", "tulips", "Tyres", "Wallet", "washing machine", "Wrist Watch", "writing pad"};
        Ger = new String[]{"klimaanlage", "airpods", "badminton schlager", "badminton shuttle", "bandagen", "baseball schlager", "stizsack", "gurtel", "fahrrad", "flasche", "krawatte", "box", "birne", "kalender", "deckenventilayor", "zellenbatterie", "stuhl", "ladegerat", "uhr", "kamm", "computer tastatur", "computermaus", "computer RAM", "zentralprozessor", "cricket schlager", "vorhange", "kissen", "ganseblumchen", "lowenzahn", "digital kamera", "digitale waage", "diode", "hunde", "tur", "turmatte", "mulltonne", "dvd-disc", "dvd spieler", "verlangerungskabel", "schutzmaske", "lufterkondensator", "diskette", "gabel", "formelles hemd", "gaszylinder", "jeans", "messer", "laptop", "speicherkarte", "mikrowelle", "hauptplatine", "tassen", "krawatte", "stift", "bleistift", "rasierer", "kuhlschrank", "relaisschaltung", "rose", "rahmen", "schere", "schuhe", "sim karte", "socken", "lautsprecher", "brille", "loffel", "sonnenblume", "tischlufter", "tischlampe", "fernsehen", "zahnburste", "transistor", "trimmer", "trophae", "t-shirt", "leuchtstoffrohre", "tulpen", "reifen", "brieftasche", "washmaschine", "armbanduhr", "schreibblock"};
        Chi = new String[]{"lengqi ji", "fei jiao", "yumaoqiu pai", "yumaoqiu chuansuo", "bengdai", "bangqiu bang", "dou dai", "yaodai", "zixingche", "pingzi", "lingjie", "hezi", "diandengpao", "rili", "diaoshan", "dianchi", "yizi", "chongdian qi", "zhong", "shuzi", "jisuanji jianpan", "diannao shubiao", "diannao neicun", "zhongyang chuli qi", "ban qiupai", "Chuānglián", "Diànzi", "Chújú", "Púgōngyīng", "Shùmǎ xiàngjī", "Shùzì chèng", "Èrjíguǎn", "Gǒu", "Mén", "Mén diàn", "Lèsè xiāng", "DVD guāngpán", "DVD bòfàng jī", "Yáncháng xiàn", "Kǒuzhào", "Fēngshàn diànróng", "Ruǎnpán", "Chā", "Zhèngshì chènshān", "Qì píng", "Niúzǎikù", "Dāo", "Bǐjìběn diànnǎo", "Cúnchú kǎ", "Wéibō", "Mǔ bǎn", "Mǎkè bēi", "Lǐngjié", "Bǐ", "Qiānbǐ", "Tìdāo", "Bīngxiāng", "Jìdiànqì diànlù", "Méiguī", "Guīmó", "Jiǎnz", "Táidēngi", "Xié", "SIM kǎ", "Wàzi", "Yángshēngqì", "Yǎnjìng", "Sháozi", "Xiàngrìkuí", "Tái shàn", "Táidēng", "Diànshì", "Yáshuā", "Jīngtǐguǎn", "Wéitiáo", "Bēi", "T xù", "Dēng guǎn", "Yùjīnxiāng", "Tāi", "Qiánbāo", "Xǐyījī", "Shǒubiǎo", "Xiězì bǎn"};
        Jap = new String[]{"eakon", "eapoddo", "badmintonraketto", "badmintonshatoru", "hotai", "yakyu-yo batto", "mame-bukuro", "beruto", "jitensha", "botoru", "chonekutai", "bokkusu", "barubu", "karenda", "tenjo fan", "serubatteri", "isu", "juden-ki", "tokei", "kushi", "konpyuta kibodo", "konpyuta no mausu", "konpyutaramu", "cpu", "kurikettobatto", "katen", "kusshon", "deiji", "tanpopo", "dejitaru kamera", "dejitarusukeru", "daiodo", "inu", "doa", "doamatto", "gomibako", "dvd disuku", "dvd pureya", "encho kodo", "feisumasuku", "fankondensa", "furoppidisuku", "foku", "fomarushatsu", "gasubonbe", "jinzu", "naifu", "rappu toppu", "memorikado", "denjirenji", "mazabodo", "magukappu", "nekutai", "pen", "enpitsu", "kami sori", "reizoko", "rire kairo", "rozu", "kibo", "hasami", "kutsu", "shimu kado", "kutsushita", "supika", "megane", "supun", "himawari", "takujo senpuki", "denki sutando", "terebi", "haburashi", "toranjisuta", "torima", "torofi", "tishatsu", "chuburaito", "churippu", "taiya", "saifu", "sentakki", "udedokei", "memocho"};
        Ita = new String[]{"condizionatore", "airpods", "racchetta di badminton", "navetta per badminton", "bende", "Mazza da baseball", "sacchetto di fagioli", "cintura", "bicicletta", "bottiglia", "cravatta a farfalla", "scatola", "lampadina", "calendario", "ventilatore", "batteria cellulare", "sedia", "caricabatterie", "orologio", "pettin", "tastiera del computer", "mouse del computer", "ram del computer", "processore", "Mazza da cricket", "le tende", "cuscino", "margherita", "dente di leone", "Camera digitale", "bilancia digitale", "diodo", "cagna", "porta", "zerbino", "pattumiera", "disco dvd", "lettore DVD", "prolunga", "mascherina", "condensatore del ventilatore", "floppy disc", "forchetta", "camicia formale", "cilindro del gas", "jeans", "coltello", "il computer portatile", "scheda di memoria", "microonde", "scheda madre", "tazza", "cravatta al collo", "penna", "matita", "rasoio", "frigorifero", "circuito relè", "rosa", "scala", "forbice", "scarpe", "carta SIM", "calzini", "altoparlante", "spettacoli", "cucchiaio", "girasole", "ventilatore da tavolo", "lampada da tavolo", "televisione", "spazzolino", "transistor", "trimmer", "trofeo", "maglietta", "luce al neon", "tulipani", "pneumatico", "portafoglio", "lavatrice", "orologio da polso", "blocco per scrivere"};
        Kor = new String[]{"eeokeon", "eeo pas", "baedeuminteon lakes", "baedeuminteon syeoteul", "bungdae", "yagu bangmang-i", "kong jumeoni", "belteu", "jajeongeo", "byeong", "nabi negtai", "sangja", "gugeun", "dallyeog", "cheonjang seonpung-gi", "sel baeteoli", "uija", "chungjeongi", "sigye", "bis", "keompyuteo kibodeu", "keompyuteo mauseu", "keompyuteo laem", "cpu", "keulikes bangmang-i", "keoteun", "bangseog", "deiji", "mindeulle", "dijiteol kamelo", "dijiteol seukeil", "daiodeu", "gae", "mun", "do-eo maeteu", "sseuleogitong", "dvd diseukeu", "dvd peulleieo", "yeonjang kodeu", "maseukeu", "paen keopaesiteo", "peullopi diseukeu", "pokeu", "jeongjang syeocheu", "giche sillindeo", "cheongbaji", "kal", "noteubug", "memoli kadeu", "maikeulopa", "madeo bodeu", "eolgul", "negtai", "pen", "yeonpil", "myeondokal", "naengjang-go", "lillei hoelo", "jangmi", "gyumo", "gawi", "sinbal", "sim kadeu", "yangmal", "seupikeo", "angyeong", "sudgalag", "haebalagi", "pyo buchae", "pyo laempeu", "tellebijeon", "chis-sol", "teulaenjiseuteo", "nakksijji", "teulopi", "tisyeocheu", "tyubeu laiteu", "tyullib", "taieo", "jigab", "setaggi", "sonmog sigye", "sseugi paedeu"};
        Fre = new String[]{"climatiseur", "airpods", "racquette de badminton", "navette de badminton", "des pansements", "batte de baseball", "sac de haricots", "courroie", "bicyclette", "bouteille", "noeud papillon", "boite", "ampoule", "calendrier", "ventilateur de plafond", "batterie cellulaire", "chaise", "chargeur", "horloge", "peigne", "clavier d'ordinateur", "souris d'ordinateur", "ordinateur ram", "cpu", "batte de cricket", "rideaux", "coussin", "marguerite", "pissenlit", "appareil photo numerique", "pese-personne numerique", "diode", "chiennes", "porte", "tapis de porte", "poubelle", "disque dvd", "lecteur de dvd", "rallonges", "masque", "condensateur de ventilateur", "disquette", "fourchette", "chemise formelle", "bouteille de gaz", "jeans", "couteau", "ordinateur portable", "carte memoire", "four micro onde", "carte mere", "tasses", "cravate au cou", "plume", "crayon", "le rasoir", "refrigerateur", "circuit relais", "rose", "escalader", "les ciseaux", "des chaussures", "carte sim", "des chaussettes", "oratrice", "lunettes", "cuillere", "tournesol", "ventilateur de table", "lampe de table", "television", "brosse a dents", "transistor", "tondeuse", "trophee", "t-shirt", "tubelight", "tulipes", "pneus", "portefeuille", "machine a laver", "montre-bracelet", "bloc-note"};
        String s=topLables[2];
        for(int i=0;i<Eng.length;i++){
            if(s.equals(Eng[i])){
                z=i;
                //Toast.makeText(Classify.this,Ger[i],Toast.LENGTH_SHORT).show();
            }
        }
        if(str1.equals("GERMAN")){
            Toast.makeText(Classify.this,Ger[z],Toast.LENGTH_SHORT).show();
            textToSpeech.speak(Ger[z],TextToSpeech.QUEUE_FLUSH,null);
        }
        else if (str1.equals("ENGLISH")){
            Toast.makeText(Classify.this,Eng[z],Toast.LENGTH_SHORT).show();
            textToSpeech.speak(Eng[z],TextToSpeech.QUEUE_FLUSH,null);
        }
        else if (str1.equals("JAPANESE")){
            Toast.makeText(Classify.this,Jap[z],Toast.LENGTH_SHORT).show();
            textToSpeech.speak(Jap[z],TextToSpeech.QUEUE_FLUSH,null);
        }
        else if (str1.equals("CHINESE")){
            Toast.makeText(Classify.this,Chi[z],Toast.LENGTH_SHORT).show();
            textToSpeech.speak(Chi[z],TextToSpeech.QUEUE_FLUSH,null);
        }
        else if (str1.equals("ITALIAN")){
            Toast.makeText(Classify.this,Ita[z],Toast.LENGTH_SHORT).show();
            textToSpeech.speak(Ita[z],TextToSpeech.QUEUE_FLUSH,null);
        }
        else if (str1.equals("KOREAN")){
            Toast.makeText(Classify.this,Kor[z],Toast.LENGTH_SHORT).show();
            textToSpeech.speak(Kor[z],TextToSpeech.QUEUE_FLUSH,null);
        }
        else {
            Toast.makeText(Classify.this,Fre[z],Toast.LENGTH_SHORT).show();
            textToSpeech.speak(Fre[z],TextToSpeech.QUEUE_FLUSH,null);
        }



    }


    // resizes bitmap to given dimensions
    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }
}