from tensorflow.keras import optimizers

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers
import tensorflow_docs.modeling as tfmodel

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow_hub as hub

partial_x_train_features=np.array([b'south pago pago victor mclaglen jon hall frances farmer olympe bradna gene lockhart douglass dumbrille francis ford ben welden abner biberman pedro cordoba rudy robles bobby stone nellie duran james flavin nina campana alfred e green treasure hunt adventure adventure',
 b'easy virtue jessica biel ben barnes kristin scott thomas colin firth kimberley nixon katherine parkinson kris marshall christian brassington charlotte riley jim mcmanus pip torrens jeremy hooton joanna bacon maggie hickey georgie glen stephan elliott young englishman marry glamorous american brings home meet parent arrive like blast future blow entrenched british stuffiness window comedy romance',
 b'fragments antonin gregori derangere anouk grinberg aurelien recoing niels arestrup yann collette laure duthilleul david assaraf pascal demolon jean baptiste iera richard sammel vincent crouzet fred epaud pascal elso nicolas giraud michael abiteboul gabriel le bomin psychiatrist probe mind traumatized soldier attempt unlock secret drove gentle deeply disturbed world war veteran edge insanity drama war',
 b'milka film taboos milka elokuva tabuista irma huntus leena suomu matti turunen eikka lehtonen esa niemela sirkka metsasaari tauno lehtihalmes ulla tapaninen toivo tuomainen hellin auvinen salmi rauni mollberg small finnish lapland community milka innocent year old girl live mother miss dead father prays god love haymaking employ drama',
 b'sleeping car david naughton judie aronson kevin mccarthy jeff conaway dani minnick ernestine mercer john carl buechler gary brockette steve lundquist billy stevenson michael scott bicknell david coburn nicole hansen tiffany million robert ruth douglas curtis jason david naughton move abandon train car resurrect vicious ghost landlady dead husband mister near fatal encounter comedy horror'])

partial_x_train_plot=np.array([b'treasure hunt adventure',
 b'young englishman marry glamorous american brings home meet parent arrive like blast future blow entrenched british stuffiness window',
 b'psychiatrist probe mind traumatized soldier attempt unlock secret drove gentle deeply disturbed world war veteran edge insanity',
 b'small finnish lapland community milka innocent year old girl live mother miss dead father prays god love haymaking employ',
 b'jason david naughton move abandon train car resurrect vicious ghost landlady dead husband mister near fatal encounter'])

partial_x_train_actors_array=np.array([np.array([b'victor mclaglen', b'jon hall', b'frances farmer',
       b'olympe bradna', b'gene lockhart', b'douglass dumbrille',
       b'francis ford', b'ben welden', b'abner biberman',
       b'pedro de cordoba', b'rudy robles', b'bobby stone',
       b'nellie duran', b'james flavin', b'nina campana'], dtype='|S18'),
np.array([b'jessica biel', b'ben barnes', b'kristin scott thomas',
       b'colin firth', b'kimberley nixon', b'katherine parkinson',
       b'kris marshall', b'christian brassington', b'charlotte riley',
       b'jim mcmanus', b'pip torrens', b'jeremy hooton', b'joanna bacon',
       b'maggie hickey', b'georgie glen'], dtype='|S21'),
np.array([b'gregori derangere', b'anouk grinberg', b'aurelien recoing',
       b'niels arestrup', b'yann collette', b'laure duthilleul',
       b'david assaraf', b'pascal demolon', b'jean-baptiste iera',
       b'richard sammel', b'vincent crouzet', b'fred epaud',
       b'pascal elso', b'nicolas giraud', b'michael abiteboul'],
      dtype='|S18'),
np.array([b'irma huntus', b'leena suomu', b'matti turunen',
       b'eikka lehtonen', b'esa niemela', b'sirkka metsasaari',
       b'tauno lehtihalmes', b'ulla tapaninen', b'toivo tuomainen',
       b'hellin auvinen-salmi'], dtype='|S20'),
np.array([b'david naughton', b'judie aronson', b'kevin mccarthy',
       b'jeff conaway', b'dani minnick', b'ernestine mercer',
       b'john carl buechler', b'gary brockette', b'steve lundquist',
       b'billy stevenson', b'michael scott-bicknell', b'david coburn',
       b'nicole hansen', b'tiffany million', b'robert ruth'], dtype='|S22')], dtype=object)

partial_x_train_reviews=np.array([b'edward small take director alfred e green cast crew uncommonly attractive brilliant assemblage south sea majority curiously undersung piece location far stylize date goldwyn hurricane admittedly riddle cliche formula package visual technical excellence scarcely matter scene stop heart chiseled adonis jon hall porcelain idol frances farmer outline profile s steam background volcano romantic closeup level defies comparison edward small film typically string frame individual work art say outdid do workhorse composer edward ward song score year prior work universal stun phantom opera',
 b'jessica biel probably best know virtuous good girl preacher kid mary camden heaven get tackle classic noel coward role early play easy virtue american interloper english aristocratic family unsettle family matriarch kristin scott thomas noel coward write upper class twit pretension wit keep come kind adopt way adopt oscar wilde george bernard shaw kid grow poverty way talent entertain upper class take coward heart felt modern progressive generally term social trend whittakers easy virtue kind aristocrat anybody like hang party invite noel entertain amelia earhart aviation jessica biel character auto race young widow detroit area course area motor car auto race fresh win monte carlo win young ben barnes heir whittaker estates lot land debt barnes bring biel home family mortify classless american way sense recognize class distinction thing get rid title nobility aristocrats story scott thomas dominate family try desperately estate husband colin firth serve world war horror do probably horror trench war slaughter fact class distinction tend melt combat biel kind like wife rule whittaker roost scandal past threatens disrupt barnes biel marriage form crux story turn fact end really viewer figure eventually happen second film adaption easy virtue silent film direct young alfred hitchcock easy virtue actually premier america london star great american stage actress jane cowl guess coward figure american heroine best american theatergoer british one version easy virtue direct flawlessly stephen elliot fine use period music noel coward cole porter end credit really mock upper class coward tradition play going gets tough tough going believe elliott try say class especially one right stuff course obligatory fox hunt upper class indulge oscar wilde say unspeakable uneatable chance younger generation expose noel coward worth see',
 b'saw night eurocine event movie european country show day european city hear le bomin barely hear derangere la chambre des officiers fortunately surprise discover great talent unknown large audience derangere absolutely astonish play character antonin verset victim post wwi trauma live trouble scene endure month war cast excellent great work cinematography offer really nice shot great landscape stun face edit really subtile bit memory make sense story minute movie show real chill ww archive action flick like sensitive psychologic movie really think absolutely recommend les fragments d antonin let le bomin',
 b'rauni mollberg earth sinful song favorite foreign film establish director major talent film festival circuit get amazing followup milka base work novelist timo mukka till worthy major dvd exposure unlike kaurismaki bros follow double handedly create tongue cheek deadpan finnish film style fan world mollberg commit naturalistic approach film overflow nature life lust earthiness find scandi cinema mainly work famous talent swede vilgot sjoman curious yellow fame director film tabu title imply mollberg effort quite effective sidestep fully treat screen theme incest making adult character father figure real blood relate daddy applies usual merely step father gimmick use countless time american movie incest work matti turunen kristus perkele translate christ devil really common law step dad underage milka beautiful offbeat fashion young girl portray shot irma huntus bring screen sexiness bergman harriet andersson decade earlier create international success summer monika sawdust tinsel imagine actress milka role shame do pursue act career afterward completing strong line leena suomu earth mother type confines act narrow emotional range prove solid rock crucial role bookended spectacularly beautiful shot birch wood winter virtually black white visually color presence milka film quickly develop nature theme presence strange click beak bird talisman early scene milka handyman turunen frolicking naked lake emerge oh natural sex play year old milka man result tastefully shoot intimacy imply ejaculation set trouble come religious aspect remote farm community heavily stress especially enjoy motif spiritual guidance cantor malmstrom quality anti stereotypical play eikka lehtonen instead rigid cruel turn care milka illegitimate baby bear strong romance turunen stud continue service mom woman neighborhood present utterly natural viewer position watch ethnographic exercise moralistic tale powerful technique milka frequently speak directly camera viewer forceful monologue bear crisp sound record sound nature include rain constant motif make milka engross experience view film subtitle knowledge finnish lapp recall best silent era classic direction strong convey dramatic content theme way transcend language kudos mollberg talented cinematographer job work remain obscurity ripe rediscovery',
 b'wonder horror film write woody allen wannabe come like check imaginatively direct typical enjoyable haunt place premise solid makeup effect good job major flaw dialogue overload cheeky wisecrack witticisms sample want scary shopping ex wife hit mark deliver inappropriate moment hero battle evil ghost'])

partial_y_train=[[0,1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]] #multilabel classification
partial_y_train=np.asarray(partial_y_train)

# Test variables
x_val_features=np.array([b'corman world exploits hollywood rebel paul w s anderson allan arkush eric balfour paul bartel peter bogdanovich bob burns david carradine gene corman julie corman roger corman joe dante jonathan demme robert niro bruce dern frances doel alex stapleton documentary diy producer director roger corman alternative approach make movie hollywood documentary',
 b'blood bone michael jai white julian sands eamonn walker dante basco nona gaye michelle belegrin bob sapp dick anthony williams francis capra ron yuan kevin kimbo slice ferguson gina carano maurice smith ernest miller kevin phillips ben ramsey los angeles ex take underground fight world storm quest fulfill promise dead friend action drama',
 b'slender man joey king julia goldani telles jaz sinclair annalise basso alex fitzalan taylor richardson javier botet jessica blank michael reilly burke kevin chapman miguel nascimento eddie frateschi oscar wahlberg danny beaton gabrielle lorthe sylvain white small town massachusetts group friend fascinate internet lore slender man attempt prove do actually exist mysteriously go miss horror',
 b'friend ivan lapshin moy drug ivan lapshin andrei boltnev nina ruslanova andrey mironov aleksey zharkov zinaida adamovich aleksandr filippenko yuriy kuznetsov valeriy filonov anatoly slivnikov andrey dudarenko semyon farada nina usatova valeri kuzin lidiya volkova yuri aroyan aleksey german russian provincial town middle stalin great purge ivan lapshin head local police do do drama',
 b'noon till charles bronson jill ireland douglas fowley stan haze damon douglas hector morales bert williams davis roberts betty cole william lanteau larry french michael leclair anne ramsey howard brunner don red barry frank d gilroy spending unforgettable hour outlaw beautiful young widow turn story worldwide famous comedy romance western'])

x_val_plot=np.array([b'documentary diy producer director roger corman alternative approach make movie hollywood',
 b'los angeles ex take underground fight world storm quest fulfill promise dead friend',
 b'small town massachusetts group friend fascinate internet lore slender man attempt prove do actually exist mysteriously go miss',
 b'russian provincial town middle stalin great purge ivan lapshin head local police do do',
 b'spending unforgettable hour outlaw beautiful young widow turn story worldwide famous'])

x_val_actors_array=np.array([np.array([b'paul w.s. anderson', b'allan arkush', b'eric balfour',
       b'paul bartel', b'peter bogdanovich', b'bob burns',
       b'david carradine', b'gene corman', b'julie corman',
       b'roger corman', b'joe dante', b'jonathan demme',
       b'robert de niro', b'bruce dern', b'frances doel'], dtype='|S18'),
 np.array([b'michael jai white', b'julian sands', b'eamonn walker',
       b'dante basco', b'nona gaye', b'michelle belegrin', b'bob sapp',
       b'dick anthony williams', b'francis capra', b'ron yuan',
       b"kevin 'kimbo slice' ferguson", b'gina carano', b'maurice smith',
       b'ernest miller', b'kevin phillips'], dtype='|S28'),
 np.array([b'joey king', b'julia goldani telles', b'jaz sinclair',
       b'annalise basso', b'alex fitzalan', b'taylor richardson',
       b'javier botet', b'jessica blank', b'michael reilly burke',
       b'kevin chapman', b'miguel nascimento', b'eddie frateschi',
       b'oscar wahlberg', b'danny beaton', b'gabrielle lorthe'],
      dtype='|S20'),
 np.array([b'andrei boltnev', b'nina ruslanova', b'andrey mironov',
       b'aleksey zharkov', b'zinaida adamovich', b'aleksandr filippenko',
       b'yuriy kuznetsov', b'valeriy filonov', b'anatoly slivnikov',
       b'andrey dudarenko', b'semyon farada', b'nina usatova',
       b'valeri kuzin', b'lidiya volkova', b'yuri aroyan'], dtype='|S20'),
 np.array([b'charles bronson', b'jill ireland', b'douglas fowley',
       b'stan haze', b'damon douglas', b'hector morales',
       b'bert williams', b'davis roberts', b'betty cole',
       b'william lanteau', b'larry french', b'michael leclair',
       b'anne ramsey', b'howard brunner', b"don 'red' barry"],
      dtype='|S15')], dtype=object)

x_val_reviews=np.array([b'sam kinison brilliant vulgar obscene use x rated material verbiage express comedic commentary variety topic funny thing tell truth superb laugh film win award best direction best documentary deservedly larry carroll do commendable job interweave live performance peer commentary narrative beverly d angelo richard pryor particularly effective brother kinison executive producer partial host insightful film man finally place let rip',
 b'theme movie tracks dennis hopper rolling thunder devane american home public general idea horror viet nam war bring war home people home understand violence nam strong stuff film depicts homecoming combat weary u s army green berets rent car travel country meet different people pick accidentally kill girl watch high school basketball game totally disillusion come home end similar dennis hopper film tracks tracks freeze frame violent end welcome home soldier boys take graphic conclusion end scene blood craze vet freeze frame leave lasting impact raw brutal beat act intense fan symbolism joe don baker viet nam era movie movie',
 b'loved great comedy know movie guy watch think funny dark performance great think write looked variey review say vibrant snappy surprisingly fresh better living breathe little life increasingly number genre dysfunctional family comedy check variety review want great opinion credible source industry p s archive retrieve review',
 b'longtime classic film buff great come worthwhile film hollywood golden age know exist see doubly nice don ameche film year immediately follow departure fox think better light comedian movie expensively mount romantic comedy family comedy show beautiful new print tcm sets cinematography elaborate idiom life father myrna loy despite reviewer say lubitsch heaven wait good ameche loy masterful job light comedy role ignore old part play loy easily manged sexy charm beautiful despite handicap overly heavy make used entire film obviously hide probably time',
 b'walked movie expect completely different get people use excuse hate movie like act excellent justin chatwin margarita levieva incredibly believable really enjoy material understand people mad people expect teenage horror flick glad depth beauty opinion do like do understand horror obsess teen soundtrack amaze love movie promotion mean trailers lot differently better strongly encourage movie love deep thought provoke beautiful emotional movie'])

y_val=[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]]
y_val=np.asarray(y_val)

class Callback_Configurations():
    
    MONITOR_METRIC = 'val_loss'
    MINIMUM_DELTA = 1
    PATIENCE = 5
    VERBOSE = 0
    MODE = 'min'
    
def callback(saved_model, model):
    
    weights_fname='{}.h5'.format(saved_model)

    try:
        with open('{}.json'.format(save_model),'r') as f:
            model_json = json.load(f)
        
        model = model_from_json(model_json)
        
        model.load_weights('{}').format(weights_fname)

    except:
        print('\nPre-trained weights not found. Fitting from start')
        pass

    monitor_metric = Callback_Configurations.MONITOR_METRIC
    
    callbacks = [
        tfmodel.EpochDots(),
        
        EarlyStopping(monitor=monitor_metric,
                      min_delta=Callback_Configurations.MINIMUM_DELTA,
                      patience=Callback_Configurations.PATIENCE,
                      verbose=Callback_Configurations.VERBOSE,
                      mode=Callback_Configurations.MODE,
                      restore_best_weights=True), # I get the error here: TypeError: object of type 'NoneType' has no len()

        ModelCheckpoint(filepath=weights_fname,
                        monitor=monitor_metric,
                        verbose=Callback_Configurations.VERBOSE,
                        save_best_only=True,
                        save_weights_only=True), #True, False      
]
    return callbacks

# import the pre-trained model
model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(model, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True)

# create the neural network structure
model = tf.keras.Sequential(name="English_Google_News_130GB_witout_OOV_tokens")
model.add(hub_layer)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, kernel_regularizer=regularizers.l2(0.01),
                                activation='relu'))
model.add(tf.keras.layers.Dropout(0.0))
model.add(tf.keras.layers.Dense(y_val.shape[1],  activation='sigmoid'))

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.01,
    decay_steps=int(np.ceil((len(partial_x_train_actors_array)*0.8)//16))*1000,
    decay_rate=1,
    staircase=False)

def optimizer_adam_v2():
    return keras.optimizers.Adam(lr_schedule)

optimizer = optimizer_adam_v2()

model.compile(optimizer=optimizer,
              loss="binary_crossentropy",
              metrics=["accuracy"])

history = model.fit([partial_x_train_features, partial_x_train_plot, partial_x_train_actors_array, partial_x_train_reviews],
                    partial_y_train,
                    steps_per_epoch=int(np.ceil((len(partial_x_train_actors_array)*0.8)//16)),
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    verbose=2,
                    callbacks=callback("english_google_news_without_oovtokens", model))