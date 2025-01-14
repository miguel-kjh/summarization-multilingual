import os
import re
from openai import OpenAI

class DocumentSummaryEvaluator:
    def __init__(self, api_key, prompt_files):
        """
        Initialize the evaluator.

        :param api_key: OpenAI API key.
        :param prompt_files: Dictionary mapping evaluation criteria to their respective prompt files. 
                            Example: {"coherence": "coh.txt", "consistency": "con.txt", "fluency": "flu.txt", "relevance": "rel.txt"}
        """
        self.client = OpenAI(
            api_key=api_key,  # This is the default and can be omitted
        )
        self.prompt_files = prompt_files

    def _load_prompts(self, language="spanish"):
        """
        Load prompts from text files.

        :param prompt_files: Dictionary mapping evaluation criteria to file paths.
        :return: Dictionary mapping evaluation criteria to their respective prompts.
        """
        prompts = {}
        for criteria, file_path in self.prompt_files.items():
            file_path = os.path.join("prompts", language, file_path)
            with open(file_path, 'r') as file:
                prompts[criteria] = file.read()
        return prompts

    def evaluate(self, document, summary, language="spanish"):
        """
        Evaluate a summary against a document for all criteria.

        :param document: The source document.
        :param summary: The summary to evaluate.
        :return: Dictionary with scores for each evaluation criterion.
        """
        results = {}
        prompts = self._load_prompts(language)
        for criteria, prompt in prompts.items():
            evaluation_prompt = prompt.replace("{{Document}}", document).replace("{{Summary}}", summary)
            response = self._call_api(evaluation_prompt)
            results[criteria] = self._parse_response(response)
        return results

    def _call_api(self, prompt):
        """
        Call the OpenAI API with the given prompt.

        :param prompt: The prompt to send to the API.
        :return: The API response.
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert evaluator for document summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
        )
        return response

    def _parse_response(self, response):
        """
        Parse the API response to extract the score.

        :param response: The API response.
        :return: The extracted score.
        """
        content = response.choices[0].message.content
        print(content)
        matched = re.search("^ ?([\d\.]+)", content)
        if (matched):
            try:
                score = float(matched.group(1))
            except:
                score = 0
        else:
            score = 0
        return score

# Example usage
if __name__ == "__main__":
    prompt_files = {
        "coherence": "coh.txt",
        "consistency": "con.txt",
        "fluency": "flu.txt",
        "relevance": "rel.txt"
    }
    language = "spanish"

    import json
    with open("../api/key.json", "r") as file:
        api_key = json.load(file)

    evaluator = DocumentSummaryEvaluator(api_key["key"], prompt_files)

    # Sample document and summary
    document = """
    Fondo Social Europeo
Fondo Social Europeo
 
SÍNTESIS DEL DOCUMENTO:
Reglamento (UE) 1304/2013: el Fondo Social Europeo
¿CUÁL ES EL OBJETIVO DE ESTE REGLAMENTO?
Establece los principios, normas y estándares para la aplicación del Fondo Social Europeo (FSE).  En el período 2014-2020, el FSE abarca cuatro ámbitos principales de inversión :empleo, y en particular empleo juvenil;inclusión social;educación; ybuena gobernanza (es decir, la mejora de la calidad de la Administración pública).  
PUNTOS CLAVE
Objetivos generalesEl FSE invierte en las personas con el objeto de mejorar las oportunidades de empleo y educación en toda la Unión Europea (UE). Durante el período 2014-2020, tiene por objeto prestar especial atención a los grupos vulnerables, como los jóvenes. El Reglamento describe el ámbito de actuación del FSE y su relación con la Iniciativa de Empleo Juvenil (IEJ).
Ámbitos fundamentales
El FSE se centra en varios ámbitos fundamentales, como:el fomento del empleo y el apoyo de la movilidad laboral;  el fomento de la inclusión social y la lucha contra la pobreza;  la inversión en educación, adquisición de capacidades y aprendizaje permanente;  la mejora de la capacidad institucional y la eficiencia de la Administración pública.  Regiones subvencionablesTodos los países de la UE pueden optar a la financiación del FSE. Un amplio abanico de organizaciones, incluidos los sectores público y privado, pueden solicitarla a través de los países de la UE.
Prioridades presupuestarias
Por primera vez, se propone una parte mínima de financiación para el FSE, fijada en el 23,1 % de la política de cohesión, y que corresponde a más de 80 000 millones de euros, asignada a la programación del FSE para el período 2014-2020.En cada país de la UE, como mínimo el 20 % del FSE debe destinarse a inclusión social y a combatir la pobreza. Esto supone ayuda para las personas vulnerables y los grupos desfavorecidos para que obtengan las cualificaciones y empleos que necesitan para integrarse en el mercado laboral.El FSE debe prestar ayuda específica para jóvenes complementando la IEJ con al menos 3 200 millones de euros. Esta iniciativa debe apoyar exclusivamente a jóvenes sin trabajo y no integrados en los sistemas de educación o formación de regiones cuyas tasas de desempleo juvenil sean superiores al 25%.En vista de la persistencia de tasas elevadas de desempleo juvenil en la UE, el Reglamento (UE) 2015/779 modifica el Reglamento (UE) n.o 1304/2013 y aumenta el nivel de la prefinanciación inicial adicional abonada para los programas operativos apoyados por la IEJ en 2015 del 1 % al 30 %.
Centrado en los resultadosLos programas deben centrarse en los resultados y basarse en el principio de adicionalidad*. El mecanismo de concentración (es decir, las medidas muy centradas en un determinado grupo destinatario) es importante a fin de influir de forma efectiva sobre el terreno.AplicaciónLos acuerdos de colaboración y los programas operativos acordados entre los países de la UE y la Comisión Europea establecen el marco de las inversiones estratégicas a escala nacional y regional.
Asociaciones público-privadas
El Reglamento (UE) n.o 1303/2013 dispone que el beneficiario en una asociación público-privada («APP») podrá ser una entidad de Derecho privado de un país de la UE («socio privado»). El socio privado (seleccionado para ejecutar la operación) podrá ser sustituido como beneficiario en el transcurso de la ejecución si así lo estipulan las cláusulas de la APP o el acuerdo de financiación entre dicho socio y la entidad financiera que cofinancia la operación.
El Reglamento Delegado (UE) 2015/1076 de la Comisión establece normas adicionales sobre la sustitución de un beneficiario y las responsabilidades correspondientes. En caso de que se sustituya a un beneficiario en una operación de APP financiada por los Fondos Estructurales y de Inversión Europeos (Fondos EIE), debe garantizarse que el nuevo socio u organismo preste, como mínimo, el mismo servicio y lo haga con los mismos niveles mínimos de calidad estipulados en el primer contrato de APP. Este Reglamento también establece procedimientos con respecto a propuestas de sustitución del socio privado y a la confirmación de sustitución del socio privado, así como requisitos mínimos que deben incluirse en los acuerdos APP financiados por los Fondos EIE.
¿A PARTIR DE CUÁNDO ESTÁ EN VIGOR EL REGLAMENTO?
Está en vigor desde el 21 de diciembre de 2013.
ANTECEDENTES
Para más información véase:Fondo Social Europeo (Comisión Europea)  
TÉRMINOS CLAVE
Principio de adicionalidad: la financiación del FSE no puede sustituir al gasto público de un país de la UE.
DOCUMENTO PRINCIPAL
Reglamento (UE) n.o 1304/2013 del Parlamento Europeo y del Consejo, de 17 de diciembre de 2013, relativo al Fondo Social Europeo y por el que se deroga el Reglamento (CE) n.o 1081/2006 del Consejo (DO L 347 de 20.12.2013, pp. 470-486).
Las modificaciones sucesivas del Reglamento (UE) n.o 1304/2013 se han incorporado al documento original. Esta versión consolidada solo tiene valor documental.
DOCUMENTOS CONEXOS
Reglamento Delegado (UE) 2015/1076 de la Comisión, de 28 de abril de 2015, por el que se establecen, de conformidad con el Reglamento (UE) n.o 1303/2013 del Parlamento Europeo y del Consejo, normas adicionales sobre la sustitución de un beneficiario y las responsabilidades correspondientes, y los requisitos mínimos que deberán constar en los acuerdos de asociación público-privada financiados por los Fondos Estructurales y de Inversión Europeos (DO L 175 de 4.7.2015, pp. 1-3)
Reglamento de Ejecución (UE) n.o 288/2014 de la Comisión, de 25 de febrero de 2014, que establece normas con arreglo al Reglamento (UE) n.o 1303/2013 del Parlamento Europeo y del Consejo, por el que se establecen disposiciones comunes relativas al Fondo Europeo de Desarrollo Regional, al Fondo Social Europeo, al Fondo de Cohesión, al Fondo Europeo Agrícola de Desarrollo Rural y al Fondo Europeo Marítimo y de la Pesca, y por el que se establecen disposiciones generales relativas al Fondo Europeo de Desarrollo Regional, al Fondo Social Europeo, al Fondo de Cohesión y al Fondo Europeo Marítimo y de la Pesca, en relación con el modelo para los programas operativos en el marco del objetivo de inversión en crecimiento y empleo, y con arreglo al Reglamento (UE) n.o 1299/2013 del Parlamento Europeo y del Consejo, por el que se establecen disposiciones específicas relativas al apoyo del Fondo Europeo de Desarrollo Regional al objetivo de cooperación territorial europea, en relación con el modelo para los programas de cooperación en el marco del objetivo de cooperación territorial europea (DO L 87 de 22.3.2014, pp. 1-48)
Decisión de Ejecución 2014/99/UE de la Comisión, de 18 de febrero de 2014, que establece la lista de regiones que pueden recibir financiación del Fondo Europeo de Desarrollo Regional y del Fondo Social Europeo, y de los Estados miembros que pueden recibir financiación del Fondo de Cohesión durante el período 2014-2020 (DO L 50 de 20.2.2014, pp. 22-34)
Véase la versión consolidada.
Reglamento (UE) n.o 1303/2013 del Parlamento Europeo y del Consejo, de 17 de diciembre de 2013, por el que se establecen disposiciones comunes relativas al Fondo Europeo de Desarrollo Regional, al Fondo Social Europeo, al Fondo de Cohesión, al Fondo Europeo Agrícola de Desarrollo Rural y al Fondo Europeo Marítimo y de la Pesca, y por el que se establecen disposiciones generales relativas al Fondo Europeo de Desarrollo Regional, al Fondo Social Europeo, al Fondo de Cohesión y al Fondo Europeo Marítimo y de la Pesca, y se deroga el Reglamento (CE) n.o 1083/2006 (DO L 347 de 20.12.2013, pp. 320-469)
Véase la versión consolidada.
última actualización 08.05.2018
    """
    summary = """
          Fondo Social Europeo
Fondo Social Europeo
 
SÍNTESIS DEL DOCUMENTO:
Reglamento (UE) 1304/2013: el Fondo Social Europeo
¿CUÁL ES EL OBJETIVO DE ESTE REGLAMENTO?
Establece los principios, normas y estándares para la aplicación del Fondo Social Europeo (FSE).  En el período 2014-2020, el FSE abarca cuatro ámbitos principales de inversión :empleo, y en particular empleo juvenil;inclusión social;educación; ybuena gobernanza (es decir, la mejora de la calidad de la Administración pública).  
PUNTOS CLAVE
Objetivos generalesEl FSE invierte en las personas con el objeto de mejorar las oportunidades de empleo y educación en toda la Unión Europea (UE). Durante el período 2014-2020, tiene por objeto prestar especial atención a los grupos vulnerables, como los jóvenes. El Reglamento describe el ámbito de actuación del FSE y su relación con la Iniciativa de Empleo Juvenil (IEJ).
Ámbitos fundamentales
El FSE se centra en varios ámbitos fundamentales, como:el fomento del empleo y el apoyo de la movilidad laboral;  el fomento de la inclusión social y la lucha contra la pobreza;  la inversión en educación, adquisición de capacidades y aprendizaje permanente;  la mejora de la capacidad institucional y la eficiencia de la Administración pública.  Regiones subvencionablesTodos los países de la UE pueden optar a la financiación del FSE. Un amplio abanico de organizaciones, incluidos los sectores público y privado, pueden solicitarla a través de los países de la UE.
Prioridades presupuestarias
Por primera vez, se propone una parte mínima de financiación para el FSE, fijada en el 23,1 % de la política de cohesión, y que corresponde a más de 80 000 millones de euros, asignada a la programación del FSE para el período 2014-2020.En cada país de la UE, como mínimo el 20 % del FSE debe destinarse a inclusión social y a combatir la pobreza. Esto supone ayuda para las personas vulnerables y los grupos desfavorecidos para que obtengan las cualificaciones y empleos que necesitan para integrarse en el mercado laboral.El FSE debe prestar ayuda específica para jóvenes complementando la IEJ con al menos 3 200 millones de euros. Esta iniciativa debe apoyar exclusivamente a jóvenes sin trabajo y no integrados en los sistemas de educación o formación de regiones cuyas tasas de desempleo juvenil sean superiores al 25%.En vista de la persistencia de tasas elevadas de desempleo juvenil en la UE, el Reglamento (UE) 2015/779 modifica el Reglamento (UE) n.o 1304/2013 y aumenta el nivel de la prefinanciación inicial adicional abonada para los programas operativos apoyados por la IEJ en 2015 del 1 % al 30 %.
Centrado en los resultadosLos programas deben centrarse en los resultados y basarse en el principio de adicionalidad*. El mecanismo de concentración (es decir, las medidas muy centradas en un determinado grupo destinatario) es importante a fin de influir de forma efectiva sobre el terreno.AplicaciónLos acuerdos de colaboración y los programas operativos acordados entre los países de la UE y la Comisión Europea establecen el marco de las inversiones estratégicas a escala nacional y regional.
Asociaciones público-privadas
El Reglamento (UE) n.o 1303/2013 dispone que el beneficiario en una asociación público-privada («APP») podrá ser una entidad de Derecho privado de un país de la UE («socio privado»). El socio privado (seleccionado para ejecutar la operación) podrá ser sustituido como beneficiario en el transcurso de la ejecución si así lo estipulan las cláusulas de la APP o el acuerdo de financiación entre dicho socio y la entidad financiera que cofinancia la operación.
El Reglamento Delegado (UE) 2015/1076 de la Comisión establece normas adicionales sobre la sustitución de un beneficiario y las responsabilidades correspondientes. En caso de que se sustituya a un beneficiario en una operación de APP financiada por los Fondos Estructurales y de Inversión Europeos (Fondos EIE), debe garantizarse que el nuevo socio u organismo preste, como mínimo, el mismo servicio y lo haga con los mismos niveles mínimos de calidad estipulados en el primer contrato de APP. Este Reglamento también establece procedimientos con respecto a propuestas de sustitución del socio privado y a la confirmación de sustitución del socio privado, así como requisitos mínimos que deben incluirse en los acuerdos APP financiados por los Fondos EIE.
¿A PARTIR DE CUÁNDO ESTÁ EN VIGOR EL REGLAMENTO?
Está en vigor desde el 21 de diciembre de 2013.
ANTECEDENTES
Para más información véase:Fondo Social Europeo (Comisión Europea)  
TÉRMINOS CLAVE
Principio de adicionalidad: la financiación del FSE no puede sustituir al gasto público de un país de la UE.
DOCUMENTO PRINCIPAL
Reglamento (UE) n.o 1304/2013 del Parlamento Europeo y del Consejo, de 17 de diciembre de 2013, relativo al Fondo Social Europeo y por el que se deroga el Reglamento (CE) n.o 1081/2006 del Consejo (DO L 347 de 20.12.2013, pp. 470-486).
Las modificaciones sucesivas del Reglamento (UE) n.o 1304/2013 se han incorporado al documento original. Esta versión consolidada solo tiene valor documental.
DOCUMENTOS CONEXOS
Reglamento Delegado (UE) 2015/1076 de la Comisión, de 28 de abril de 2015, por el que se establecen, de conformidad con el Reglamento (UE) n.o 1303/2013 del Parlamento Europeo y del Consejo, normas adicionales sobre la sustitución de un beneficiario y las responsabilidades correspondientes, y los requisitos mínimos que deberán constar en los acuerdos de asociación público-privada financiados por los Fondos Estructurales y de Inversión Europeos (DO L 175 de 4.7.2015, pp. 1-3)
Reglamento de Ejecución (UE) n.o 288/2014 de la Comisión, de 25 de febrero de 2014, que establece normas con arreglo al Reglamento (UE) n.o 1303/2013 del Parlamento Europeo y del Consejo, por el que se establecen disposiciones comunes relativas al Fondo Europeo de Desarrollo Regional, al Fondo Social Europeo, al Fondo de Cohesión, al Fondo Europeo Agrícola de Desarrollo Rural y al Fondo Europeo Marítimo y de la Pesca, y por el que se establecen disposiciones generales relativas al Fondo Europeo de Desarrollo Regional, al Fondo Social Europeo, al Fondo de Cohesión y al Fondo Europeo Marítimo y de la Pesca, en relación con el modelo para los programas operativos en el marco del objetivo de inversión en crecimiento y empleo, y con arreglo al Reglamento (UE) n.o 1299/2013 del Parlamento Europeo y del Consejo, por el que se establecen disposiciones específicas relativas al apoyo del Fondo Europeo de Desarrollo Regional al objetivo de cooperación territorial europea, en relación con el modelo para los programas de cooperación en el marco del objetivo de cooperación territorial europea (DO L 87 de 22.3.2014, pp. 1-48)
Decisión de Ejecución 2014/99/UE de la Comisión, de 18 de febrero de 2014, que establece la lista de regiones que pueden recibir financiación del Fondo Europeo de Desarrollo Regional y del Fondo Social Europeo, y de los Estados miembros que pueden recibir financiación del Fondo de Cohesión durante el período 2014-2020 (DO L 50 de 20.2.2014, pp. 22-34)
Véase la versión consolidada.
Reglamento (UE) n.o 1303/2013 del Parlamento Europeo y del Consejo, de 17 de diciembre de 2013, por el que se establecen disposiciones comunes relativas al Fondo Europeo de Desarrollo Regional, al Fondo Social Europeo, al Fondo de Cohesión, al Fondo Europeo Agrícola de Desarrollo Rural y al Fondo Europeo Marítimo y de la Pesca, y por el que se establecen disposiciones generales relativas al Fondo Europeo de Desarrollo Regional, al Fondo Social Europeo, al Fondo de Cohesión y al Fondo Europeo Marítimo y de la Pesca, y se deroga el Reglamento (CE) n.o 1083/2006 (DO L 347 de 20.12.2013, pp. 320-469)
Véase la versión consolidada.
última actualización 08.05.2018

"""

    results = evaluator.evaluate(document, summary, language=language)
    print(results)
